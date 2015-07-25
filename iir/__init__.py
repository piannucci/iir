import numpy as np
import scipy.weave
import mkfilter

# ContinuousIIRFilter(order, alpha, beta, gamma, dtype) constructs a filter with difference equation
#
# y[i] = gamma * sum(alpha[j] * x[i+j-order] for j in xrange(order+1)) +
#        sum(beta[j] * y[i+j-order] for j in xrange(order))
#
# and transfer function
# Y(z) = gamma * sum(alpha[j] * z**(j-order) for j in xrange(order+1)) / (1 - sum(beta[j] * z**(j-order) for j in xrange(order)))
#
# whose initial state is all zeros and whose internal state representation has type dtype.

def _iir_cpp_impl(order, alpha, beta, gamma, float=False):
    suffix = 'f' if float else ''
    impl = "for (int i=0; i<%d && i<N; i++)\n" % order
    impl += "\ty(i) = x(i)*%.10e%s;\n" % (1./gamma, suffix)
    impl += "for (int i=%d; i<N; i++)\n" % order
    impl += "\ty(i) = " + " + ".join("%.10e%s*x(i%s)" % (alpha[i], suffix, str(i-order) if i!=order else "") for i in range(order+1))
    impl += " + " + " + ".join("%.10e%s*y(i%s)" % (beta[i], suffix, str(i-order)) for i in range(order))
    impl += ";\nfor (int i=0; i<N; i++)\n\ty(i) *= %.10e%s;" % (gamma, suffix)
    return impl

def _iir_cpp_impl_continuous(order, alpha, beta, gamma, float=False):
    suffix = 'f' if float else ''
    impl = "for (int i=0; i<%d; i++)\n" % order
    impl += "{\n"
    impl += "\tx(i) = x_hist(i);\n"
    impl += "\ty(i) = y_hist(i);\n"
    impl += "}\n"
    impl += "for (int i=%d; i<N+%d; i++)\n" % (order, order)
    impl += "\ty(i) = " + " + ".join("%.10e%s*x(i%s)" % (alpha[i], suffix, str(i-order) if i!=order else "") for i in range(order+1))
    impl += " + " + " + ".join("%.10e%s*y(i%s)" % (beta[i], suffix, str(i-order)) for i in range(order))
    impl += ";\n"
    impl += "for (int i=0; i<%d; i++)\n" % order
    impl += "{\n"
    impl += "\tx_hist(i) = x(N+i);\n"
    impl += "\ty_hist(i) = y(N+i);\n"
    impl += "}\n"
    impl += "for (int i=%d; i<N+%d; i++)\n" % (order, order)
    impl += "\ty(i) *= %.10e;" % gamma
    return impl

class IIRFilter(object):
    def __init__(self, order, alpha, beta, gamma):
        self.code = _iir_cpp_impl(order, alpha, beta, gamma)
        self.codef = _iir_cpp_impl(order, alpha, beta, gamma, True)
    def __call__(self, x):
        x = np.array(x, copy=False)
        y = np.zeros_like(x)
        N = y.size
        scipy.weave.inline(self.code if x.real.dtype==np.float64 else self.codef,
                     ['N','x','y'], type_converters=scipy.weave.converters.blitz, verbose=0)
        return y

class ContinuousIIRFilter(object):
    def __init__(self, order, alpha, beta, gamma, dtype):
        single = (np.dtype(dtype).name in ['float32', 'complex64'])
        self.code = _iir_cpp_impl_continuous(order, alpha, beta, gamma, single)
        self.x_hist = np.zeros(order, dtype)
        self.y_hist = np.zeros(order, dtype)
        self.order = order
    def __call__(self, x):
        x = np.array(x, copy=False)
        N = x.size
        x = np.r_[np.empty(self.order, x.dtype), x]
        y = np.empty(N+self.order, x.dtype)
        x_hist = self.x_hist
        y_hist = self.y_hist
        scipy.weave.inline(self.code, ['N','x_hist','y_hist','x','y'],
                     type_converters=scipy.weave.converters.blitz, verbose=0)
        return y[self.order:]
    def copy(self):
        other = ContinuousIIRFilter.__new__(ContinuousIIRFilter)
        other.code = self.code
        dtype = self.x_hist.dtype
        other.x_hist = np.zeros(self.order, dtype)
        other.y_hist = np.zeros(self.order, dtype)
        other.order = self.order
        return other

def mkfilter(method, type, order, alpha, continuous=False, dtype=None):
    if not isinstance(alpha, (tuple, list)):
        alpha = (alpha,)
    args = ('-%s' % method).split(' ') + ['-%s' % type, '-o', str(order), '-l', '-a'] + [str(a) for a in alpha]
    order, alpha, beta, gamma = mkfilter.mkfilter(*args)
    if not continuous:
        return IIRFilter(order, alpha, beta, gamma)
    else:
        return ContinuousIIRFilter(order, alpha, beta, gamma, dtype)

def lowpass(freq, order=6, method='Bu', continuous=False, dtype=None):
    return mkfilter(method, 'Lp', order, freq, continuous, dtype)

def highpass(freq, order=6, method='Bu', continuous=False, dtype=None):
    return mkfilter(method, 'Hp', order, freq, continuous, dtype)
