#include <Python.h>
#include <complex>
#include <vector>
#include <string>
#include <stdint.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>

#define EPS         1e-10
#define MAXORDER    10
#define MAXPZ       512

typedef std::complex<double> dcomplex;

static void mkfilter(PyObject *args, int &order_out, double alpha_out[], double beta_out[], double &gamma_out);

static PyObject *
mkfilter_mkfilter(PyObject *self, PyObject *args)
{
    int order = 0;
    double alpha[MAXPZ], beta[MAXPZ], gamma;
    try {
        mkfilter(args, order, alpha, beta, gamma);
    } catch (const char *s) {
        PyErr_SetString(PyExc_ValueError, s);
        return NULL;
    }

    if (PyErr_Occurred())
        return NULL;

    PyObject *alphaList = PyTuple_New(order+1);
    PyObject *betaList = PyTuple_New(order);

    for (Py_ssize_t i=0; i<order; i++) {
        PyTuple_SetItem(alphaList, i, Py_BuildValue("d", alpha[i]));
        PyTuple_SetItem(betaList, i, Py_BuildValue("d", beta[i]));
    }
    PyTuple_SetItem(alphaList, order, Py_BuildValue("d", alpha[order]));

    return Py_BuildValue("iNNd", order, alphaList, betaList, gamma);
}

static PyMethodDef MkfilterMethods[] = {
    {"mkfilter",  mkfilter_mkfilter, METH_VARARGS, "Designs a filter."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initmkfilter(void)
{
    Py_InitModule("mkfilter", MkfilterMethods);
}

#define opt_be 0x00001  /* -Be  Bessel characteristic          */
#define opt_bu 0x00002  /* -Bu  Butterworth characteristic     */
#define opt_ch 0x00004  /* -Ch  Chebyshev characteristic       */
#define opt_re 0x00008  /* -Re  Resonator                      */
#define opt_pi 0x00010  /* -Pi  proportional-integral          */

#define opt_lp 0x00020  /* -Lp  lowpass                        */
#define opt_hp 0x00040  /* -Hp  highpass                       */
#define opt_bp 0x00080  /* -Bp  bandpass                       */
#define opt_bs 0x00100  /* -Bs  bandstop                       */
#define opt_ap 0x00200  /* -Ap  allpass                        */

#define opt_a  0x00400  /* -a   alpha value                    */
#define opt_l  0x00800  /* -l   just list filter parameters    */
#define opt_o  0x01000  /* -o   order of filter                */
#define opt_p  0x02000  /* -p   specified poles only           */
#define opt_w  0x04000  /* -w   don't pre-warp                 */
#define opt_z  0x08000  /* -z   use matched z-transform        */
#define opt_Z  0x10000  /* -Z   additional zero                */

template <class T> static inline T sqr(T z) { return z*z; }
static inline bool onebit(unsigned int m) { return (m != 0) && ((m & m-1) == 0); }
static inline dcomplex expj(double theta) { return exp(dcomplex(0, theta)); }

static dcomplex eval(dcomplex coeffs[], int npz, dcomplex z)
{
    /* evaluate polynomial in z, substituting for z */
    dcomplex sum = dcomplex(0.0);
    for (int i = npz; i >= 0; i--) sum = (sum * z) + coeffs[i];
    return sum;
}

static dcomplex evaluate(dcomplex topco[], int nz, dcomplex botco[], int np, dcomplex z)
{
    /* evaluate response, substituting for z */
    return eval(topco, nz, z) / eval(botco, np, z);
}

struct pzrep
{
    dcomplex poles[MAXPZ], zeros[MAXPZ];
    int numpoles, numzeros;
};

static pzrep splane, zplane;
static int order;
static double raw_alpha1, raw_alpha2, raw_alphaz;
static dcomplex dc_gain, fc_gain, hf_gain;
static unsigned int options;
static double warped_alpha1, warped_alpha2, chebrip, qfactor;
static bool infq;
static unsigned int polemask;
static double xcoeffs[MAXPZ+1], ycoeffs[MAXPZ+1];

static dcomplex bessel_poles[] =
{
    /* table produced by /usr/fisher/bessel --  N.B. only one member of each C.Conj. pair is listed */
    dcomplex(-1.00000000000e+00, 0.00000000000e+00), dcomplex(-1.10160133059e+00, 6.36009824757e-01), dcomplex(-1.32267579991e+00, 0.00000000000e+00), dcomplex(-1.04740916101e+00, 9.99264436281e-01),
    dcomplex(-1.37006783055e+00, 4.10249717494e-01), dcomplex(-9.95208764350e-01, 1.25710573945e+00), dcomplex(-1.50231627145e+00, 0.00000000000e+00), dcomplex(-1.38087732586e+00, 7.17909587627e-01),
    dcomplex(-9.57676548563e-01, 1.47112432073e+00), dcomplex(-1.57149040362e+00, 3.20896374221e-01), dcomplex(-1.38185809760e+00, 9.71471890712e-01), dcomplex(-9.30656522947e-01, 1.66186326894e+00),
    dcomplex(-1.68436817927e+00, 0.00000000000e+00), dcomplex(-1.61203876622e+00, 5.89244506931e-01), dcomplex(-1.37890321680e+00, 1.19156677780e+00), dcomplex(-9.09867780623e-01, 1.83645135304e+00),
    dcomplex(-1.75740840040e+00, 2.72867575103e-01), dcomplex(-1.63693941813e+00, 8.22795625139e-01), dcomplex(-1.37384121764e+00, 1.38835657588e+00), dcomplex(-8.92869718847e-01, 1.99832584364e+00),
    dcomplex(-1.85660050123e+00, 0.00000000000e+00), dcomplex(-1.80717053496e+00, 5.12383730575e-01), dcomplex(-1.65239648458e+00, 1.03138956698e+00), dcomplex(-1.36758830979e+00, 1.56773371224e+00),
    dcomplex(-8.78399276161e-01, 2.14980052431e+00), dcomplex(-1.92761969145e+00, 2.41623471082e-01), dcomplex(-1.84219624443e+00, 7.27257597722e-01), dcomplex(-1.66181024140e+00, 1.22110021857e+00),
    dcomplex(-1.36069227838e+00, 1.73350574267e+00), dcomplex(-8.65756901707e-01, 2.29260483098e+00),
};

static void readcmdline(PyObject *);
static unsigned int decodeoptions(const char*), optbit(const char);
static void checkoptions();
static void setdefaults();
static void compute_s(), choosepole(dcomplex), prewarp(), normalize(), compute_z_blt();
static dcomplex blt(dcomplex);
static void compute_z_mzt();
static void compute_notch(), compute_apres();
static dcomplex reflect(dcomplex);
static void compute_bpres(), add_extra_zero();
static void expandpoly(), expand(dcomplex[], int, dcomplex[]), multin(dcomplex, int, dcomplex[]);

static void mkfilter(PyObject *args, int &order_out, double alpha_out[], double beta_out[], double &gamma_out)
{
    splane = zplane = (pzrep){};
    raw_alpha1 = raw_alpha2 = raw_alphaz = 0;
    dc_gain = fc_gain = hf_gain = 0;
    warped_alpha1 = warped_alpha2 = chebrip = qfactor = 0;
    infq = 0;

    readcmdline(args);
    if (PyErr_Occurred())
        return;
    checkoptions();
    setdefaults();

    if (options & opt_re)
    {
        if (options & opt_bp) compute_bpres();    /* bandpass resonator */
        if (options & opt_bs) compute_notch();    /* bandstop resonator (notch) */
        if (options & opt_ap) compute_apres();    /* allpass resonator */
    }
    else
    {
        if (options & opt_pi)
        {
            prewarp();
            splane.poles[0] = 0.0;
            splane.zeros[0] = -2.0 * M_PI * warped_alpha1;
            splane.numpoles = splane.numzeros = 1;
        }
        else
        {
            compute_s();
            prewarp();
            normalize();
        }
        if (options & opt_z) compute_z_mzt(); else compute_z_blt();
    }
    if (options & opt_Z) add_extra_zero();
    expandpoly();

    dcomplex gain = (options & opt_pi) ? hf_gain :
        (options & opt_lp) ? dc_gain :
        (options & opt_hp) ? hf_gain :
        (options & (opt_bp | opt_ap)) ? fc_gain :
        (options & opt_bs) ? sqrt(dc_gain * hf_gain) : dcomplex(1.0);

    order_out = order;
    for (int i=0; i<zplane.numzeros+1; i++)
        alpha_out[i] = xcoeffs[i];
    for (int i=0; i<zplane.numpoles+1; i++)
        beta_out[i] = ycoeffs[i];
    gamma_out = 1./std::abs(gain);
}

static void readcmdline(PyObject *args)
{
    Py_ssize_t argc = PyTuple_Size(args);

    options = order = polemask = 0;
    Py_ssize_t ap = 0;
    while (ap < argc)
    {
        PyObject *value = PyTuple_GetItem(args, ap++);
        if (!PyString_Check(value))
            throw "Option names must be strings";
        unsigned int m = decodeoptions(PyString_AsString(value));

        if (m & opt_ch)
        {
            if (ap < argc)
                chebrip = PyFloat_AsDouble(PyTuple_GetItem(args, ap++));
            else
                throw "-Ch option requires a numeric argument";
        }
        if (m & opt_a)
        {
            if (ap < argc)
                raw_alpha1 = raw_alpha2 = PyFloat_AsDouble(PyTuple_GetItem(args, ap++));
            else
                throw "-a option requires a numeric argument";
            if (ap < argc)
            {
                value = PyTuple_GetItem(args, ap);
                if (!PyString_Check(value) || PyString_AsString(value)[0] != '-')
                {
                    raw_alpha2 = PyFloat_AsDouble(value);
                    ap++;
                }
            }
        }
        if (m & opt_Z)
        {
            if (ap < argc)
                raw_alphaz = PyFloat_AsDouble(PyTuple_GetItem(args, ap++));
            else
                throw "-Z option requires a numeric argument";
        }
        if (m & opt_o)
        {
            if (ap < argc)
                order = PyInt_AsLong(PyTuple_GetItem(args, ap++));
            else
                throw "-o option requires a numeric argument";
        }
        if (m & opt_p)
        {
            while (ap < argc)
            {
                value = PyTuple_GetItem(args, ap);
                if (PyString_Check(value) && PyString_AsString(value)[0] == '-')
                    break;
                int p = PyInt_AsLong(value);
                if (p < 0 || p > 31) p = 31; /* out-of-range value will be picked up later */
                polemask |= (1 << p);
                ap++;
                if (PyErr_Occurred())
                    break;
            }
        }
        if (m & opt_re)
        {
            if (ap < argc)
            {
                value = PyTuple_GetItem(args, ap++);
                if (PyString_Check(value) && strcmp(PyString_AsString(value), "Inf") == 0)
                    infq = true;
                else
                {
                    qfactor = PyFloat_AsDouble(value);
                    infq = false;
                }
            }
            else
                throw "-Re option requires an argument";
        }
        options |= m;
        if (PyErr_Occurred())
            return;
    }
}

static unsigned int decodeoptions(const char *s)
{
    if (*(s++) != '-') throw "Options must begin with '-'";
    unsigned int m = 0;
    if      (strcmp(s, "Be") == 0) m |= opt_be;
    else if (strcmp(s, "Bu") == 0) m |= opt_bu;
    else if (strcmp(s, "Ch") == 0) m |= opt_ch;
    else if (strcmp(s, "Re") == 0) m |= opt_re;
    else if (strcmp(s, "Pi") == 0) m |= opt_pi;
    else if (strcmp(s, "Lp") == 0) m |= opt_lp;
    else if (strcmp(s, "Hp") == 0) m |= opt_hp;
    else if (strcmp(s, "Bp") == 0) m |= opt_bp;
    else if (strcmp(s, "Bs") == 0) m |= opt_bs;
    else if (strcmp(s, "Ap") == 0) m |= opt_ap;
    else
    {
        while (*s != '\0')
        {
            unsigned int bit = optbit(*(s++));
            if (bit == 0)
                throw "Unrecognized option";
            m |= bit;
        }
    }
    return m;
}

static unsigned int optbit(char c)
{
    switch (c)
    {
        default:    return 0;
        case 'a':   return opt_a;
        case 'l':   return opt_l;
        case 'o':   return opt_o;
        case 'p':   return opt_p;
        case 'w':   return opt_w;
        case 'z':   return opt_z;
        case 'Z':   return opt_Z;
    }
}

static void checkoptions()
{
    if (!onebit(options & (opt_be | opt_bu | opt_ch | opt_re | opt_pi)))
        throw "must specify exactly one of -Be, -Bu, -Ch, -Re, -Pi";
    if (options & opt_re)
    {
        if (!onebit(options & (opt_bp | opt_bs | opt_ap)))
            throw "must specify exactly one of -Bp, -Bs, -Ap with -Re";
        if (options & (opt_lp | opt_hp | opt_o | opt_p | opt_w | opt_z))
            throw "can't use -Lp, -Hp, -o, -p, -w, -z with -Re";
    }
    else if (options & opt_pi)
    {
        if (options & (opt_lp | opt_hp | opt_bp | opt_bs | opt_ap))
            throw "-Lp, -Hp, -Bp, -Bs, -Ap illegal in conjunction with -Pi";
        if (!(options & opt_o) || (order != 1)) throw "-Pi implies -o 1";
    }
    else
    {
        if (!onebit(options & (opt_lp | opt_hp | opt_bp | opt_bs)))
            throw "must specify exactly one of -Lp, -Hp, -Bp, -Bs";
        if (options & opt_ap) throw "-Ap implies -Re";
        if (options & opt_o)
        {
            if (order < 1 || order > MAXORDER) throw "order must be in range 1 .. MAXORDER";
            if (options & opt_p)
            {
                unsigned int m = (1 << order) - 1; /* "order" bits set */
                if ((polemask & ~m) != 0)
                    throw "args to -p must be in range 0 .. order-1";
            }
        }
        else throw "must specify -o";
    }
    if (!(options & opt_a)) throw "must specify -a";
}

static void setdefaults()
{
    if (!(options & opt_p)) polemask = (uint32_t)~0; /* use all poles */
    if (!(options & (opt_bp | opt_bs))) raw_alpha2 = raw_alpha1;
}

static void compute_s() /* compute S-plane poles for prototype LP filter */
{
    splane.numpoles = 0;
    if (options & opt_be)
    {
        /* Bessel filter */
        int p = (order*order)/4; /* ptr into table */
        if (order & 1) choosepole(bessel_poles[p++]);
        for (int i = 0; i < order/2; i++)
        {
            choosepole(bessel_poles[p]);
            choosepole(std::conj(bessel_poles[p]));
            p++;
        }
    }
    if (options & (opt_bu | opt_ch))
    {
        /* Butterworth filter */
        for (int i = 0; i < 2*order; i++)
        {
            double theta = (order & 1) ? (i*M_PI) / order : ((i+0.5)*M_PI) / order;
            choosepole(expj(theta));
        }
    }
    if (options & opt_ch)
    {
        /* modify for Chebyshev (p. 136 DeFatta et al.) */
        if (chebrip >= 0.0)
        {
            throw "Chebyshev ripple must be < 0.0";
        }
        double rip = pow(10.0, -chebrip / 10.0);
        double eps = sqrt(rip - 1.0);
        double y = asinh(1.0 / eps) / (double) order;
        if (y <= 0.0)
        {
            throw "bug: Chebyshev y must be > 0.0";
        }
        double sh = sinh(y);
        double ch = cosh(y);
        for (int i = 0; i < splane.numpoles; i++)
            splane.poles[i] = dcomplex(splane.poles[i].real() * sh, splane.poles[i].imag() * ch);
    }
}

static void choosepole(dcomplex z)
{
    if (z.real() < 0.0)
    {
        if (polemask & 1) splane.poles[splane.numpoles++] = z;
        polemask >>= 1;
    }
}

static void prewarp()
{
    /* for bilinear transform, perform pre-warp on alpha values */
    if (options & (opt_w | opt_z))
    {
        warped_alpha1 = raw_alpha1;
        warped_alpha2 = raw_alpha2;
    }
    else
    {
        warped_alpha1 = tan(M_PI * raw_alpha1) / M_PI;
        warped_alpha2 = tan(M_PI * raw_alpha2) / M_PI;
    }
}

static void normalize()  /* called for trad, not for -Re or -Pi */
{
    double w1 = 2.0 * M_PI * warped_alpha1;
    double w2 = 2.0 * M_PI * warped_alpha2;
    /* transform prototype into appropriate filter type (lp/hp/bp/bs) */
    switch (options & (opt_lp | opt_hp | opt_bp| opt_bs))
    {
        case opt_lp:
            {
                for (int i = 0; i < splane.numpoles; i++) splane.poles[i] = splane.poles[i] * w1;
                splane.numzeros = 0;
                break;
            }

        case opt_hp:
            {
                int i;
                for (i=0; i < splane.numpoles; i++) splane.poles[i] = w1 / splane.poles[i];
                for (i=0; i < splane.numpoles; i++) splane.zeros[i] = 0.0;  /* also N zeros at (0,0) */
                splane.numzeros = splane.numpoles;
                break;
            }

        case opt_bp:
            {
                double w0 = sqrt(w1*w2), bw = w2-w1; int i;
                for (i=0; i < splane.numpoles; i++)
                {
                    dcomplex hba = 0.5 * (splane.poles[i] * bw);
                    dcomplex temp = sqrt(1.0 - sqr(w0 / hba));
                    splane.poles[i] = hba * (1.0 + temp);
                    splane.poles[splane.numpoles+i] = hba * (1.0 - temp);
                }
                for (i=0; i < splane.numpoles; i++) splane.zeros[i] = 0.0;  /* also N zeros at (0,0) */
                splane.numzeros = splane.numpoles;
                splane.numpoles *= 2;
                break;
            }

        case opt_bs:
            {
                double w0 = sqrt(w1*w2), bw = w2-w1; int i;
                for (i=0; i < splane.numpoles; i++)
                {
                    dcomplex hba = 0.5 * (bw / splane.poles[i]);
                    dcomplex temp = sqrt(1.0 - sqr(w0 / hba));
                    splane.poles[i] = hba * (1.0 + temp);
                    splane.poles[splane.numpoles+i] = hba * (1.0 - temp);
                }
                for (i=0; i < splane.numpoles; i++)    /* also 2N zeros at (0, +-w0) */
                {
                    splane.zeros[i] = dcomplex(0.0, +w0);
                    splane.zeros[splane.numpoles+i] = dcomplex(0.0, -w0);
                }
                splane.numpoles *= 2;
                splane.numzeros = splane.numpoles;
                break;
            }
    }
}

static void compute_z_blt() /* given S-plane poles & zeros, compute Z-plane poles & zeros, by bilinear transform */
{
    int i;
    zplane.numpoles = splane.numpoles;
    zplane.numzeros = splane.numzeros;
    for (i=0; i < zplane.numpoles; i++) zplane.poles[i] = blt(splane.poles[i]);
    for (i=0; i < zplane.numzeros; i++) zplane.zeros[i] = blt(splane.zeros[i]);
    while (zplane.numzeros < zplane.numpoles) zplane.zeros[zplane.numzeros++] = -1.0;
}

static dcomplex blt(dcomplex pz)
{
    return (2.0 + pz) / (2.0 - pz);
}

static void compute_z_mzt() /* given S-plane poles & zeros, compute Z-plane poles & zeros, by matched z-transform */
{
    int i;
    zplane.numpoles = splane.numpoles;
    zplane.numzeros = splane.numzeros;
    for (i=0; i < zplane.numpoles; i++) zplane.poles[i] = exp(splane.poles[i]);
    for (i=0; i < zplane.numzeros; i++) zplane.zeros[i] = exp(splane.zeros[i]);
}

static void compute_notch()
{
    /* compute Z-plane pole & zero positions for bandstop resonator (notch filter) */
    compute_bpres();  /* iterate to place poles */
    double theta = 2.0 * M_PI * raw_alpha1;
    dcomplex zz = expj(theta); /* place zeros exactly */
    zplane.zeros[0] = zz; zplane.zeros[1] = std::conj(zz);
}

static void compute_apres()
{
    /* compute Z-plane pole & zero positions for allpass resonator */
    compute_bpres();  /* iterate to place poles */
    zplane.zeros[0] = reflect(zplane.poles[0]);
    zplane.zeros[1] = reflect(zplane.poles[1]);
}

static dcomplex reflect(dcomplex z)
{
    double r = std::abs(z);
    return z / (r*r);
}

static void compute_bpres()
{
    /* compute Z-plane pole & zero positions for bandpass resonator */
    zplane.numpoles = zplane.numzeros = 2;
    zplane.zeros[0] = 1.0; zplane.zeros[1] = -1.0;
    double theta = 2.0 * M_PI * raw_alpha1; /* where we want the peak to be */
    if (infq)
    {
        /* oscillator */
        dcomplex zp = expj(theta);
        zplane.poles[0] = zp; zplane.poles[1] = std::conj(zp);
    }
    else
    {
        /* must iterate to find exact pole positions */
        dcomplex topcoeffs[MAXPZ+1]; expand(zplane.zeros, zplane.numzeros, topcoeffs);
        double r = exp(-theta / (2.0 * qfactor));
        double thm = theta, th1 = 0.0, th2 = M_PI;
        bool cvg = false;
        for (int i=0; i < 50 && !cvg; i++)
        {
            dcomplex zp = r * expj(thm);
            zplane.poles[0] = zp; zplane.poles[1] = std::conj(zp);
            dcomplex botcoeffs[MAXPZ+1]; expand(zplane.poles, zplane.numpoles, botcoeffs);
            dcomplex g = evaluate(topcoeffs, zplane.numzeros, botcoeffs, zplane.numpoles, expj(theta));
            double phi = g.imag() / g.real(); /* approx to atan2 */
            if (phi > 0.0) th2 = thm; else th1 = thm;
            if (fabs(phi) < EPS) cvg = true;
            thm = 0.5 * (th1+th2);
        }
        if (!cvg) fprintf(stderr, "warning: failed to converge");
    }
}

static void add_extra_zero()
{
    if (zplane.numzeros+2 > MAXPZ)
        throw "too many zeros; can't do -Z";
    double theta = 2.0 * M_PI * raw_alphaz;
    dcomplex zz = expj(theta);
    zplane.zeros[zplane.numzeros++] = zz;
    zplane.zeros[zplane.numzeros++] = std::conj(zz);
    while (zplane.numpoles < zplane.numzeros) zplane.poles[zplane.numpoles++] = 0.0;  /* ensure causality */
}

static void expandpoly() /* given Z-plane poles & zeros, compute top & bot polynomials in Z, and then recurrence relation */
{
    dcomplex topcoeffs[MAXPZ+1], botcoeffs[MAXPZ+1]; int i;
    expand(zplane.zeros, zplane.numzeros, topcoeffs);
    expand(zplane.poles, zplane.numpoles, botcoeffs);
    dc_gain = evaluate(topcoeffs, zplane.numzeros, botcoeffs, zplane.numpoles, 1.0);
    double theta = M_PI * (raw_alpha1 + raw_alpha2); /* "jwT" for centre freq. */
    fc_gain = evaluate(topcoeffs, zplane.numzeros, botcoeffs, zplane.numpoles, expj(theta));
    hf_gain = evaluate(topcoeffs, zplane.numzeros, botcoeffs, zplane.numpoles, -1.0);
    for (i = 0; i <= zplane.numzeros; i++) xcoeffs[i] = +(topcoeffs[i].real() / botcoeffs[zplane.numpoles].real());
    for (i = 0; i <= zplane.numpoles; i++) ycoeffs[i] = -(botcoeffs[i].real() / botcoeffs[zplane.numpoles].real());
}

static void expand(dcomplex pz[], int npz, dcomplex coeffs[])
{
    /* compute product of poles or zeros as a polynomial of z */
    int i;
    coeffs[0] = 1.0;
    for (i=0; i < npz; i++) coeffs[i+1] = 0.0;
    for (i=0; i < npz; i++) multin(pz[i], npz, coeffs);
    /* check computed coeffs of z^k are all real */
    for (i=0; i < npz+1; i++)
    {
        if (fabs(coeffs[i].imag()) > EPS)
            throw "coeff is not real; poles/zeros are not complex conjugates";
    }
}

static void multin(dcomplex w, int npz, dcomplex coeffs[])
{
    /* multiply factor (z-w) into coeffs */
    dcomplex nw = -w;
    for (int i = npz; i >= 1; i--) coeffs[i] = (nw * coeffs[i]) + coeffs[i-1];
    coeffs[0] = nw * coeffs[0];
}
