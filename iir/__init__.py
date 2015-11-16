# The MIT License (MIT)
# 
# Copyright (c) 2015 Peter Iannucci
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import weave
from . import mkfilter

defaultChebyshevRipple_dB = -.021

# IIRFilter(order, alpha, beta, gamma, dtype) constructs a filter with difference equation
#
# y[i] = gamma * sum(alpha[j] * x[i+j-order] for j in range(order+1)) +
#        sum(beta[j] * y[i+j-order] for j in range(order))
#
# and transfer function
# Y(z) = gamma * sum(alpha[j] * z**(j-order) for j in range(order+1)) / (1 - sum(beta[j] * z**(j-order) for j in range(order)))
#
# whose initial state is all zeros and whose internal state representation has type dtype.

class IIRFilter(object):
    def __init__(self, order, alpha, beta, gamma):
        dtype = np.complex128
        kernel = " + ".join(["{coeff}*x(i+{offset})".format(coeff=float.hex(alpha[i]), i=i, offset=i-order) for i in range(order+1)] +
                            ["{coeff}*y(i+{offset})".format(coeff=float.hex(beta[i]), i=i, offset=i-order) for i in range(order)])
        self.code = """
        for (int i=0; i<{order}; i++)
        {{
            x(i) = x_hist(i);
            y(i) = y_hist(i);
        }}
        for (int i={order}; i<N+{order}; i++)
            y(i) = {kernel};
        for (int i=0; i<{order}; i++)
        {{
            x_hist(i) = x(N+i);
            y_hist(i) = y(N+i);
        }}
        for (int i={order}; i<N+{order}; i++)
            y(i) *= {gamma};
        """.format(order=order, kernel=kernel, gamma=float.hex(gamma))
        self.x_hist = np.zeros(order, dtype)
        self.y_hist = np.zeros(order, dtype)
        self.order = order
        self([dtype(0)])
    def __call__(self, x):
        x = np.array(x, copy=False)
        noncomplex = not np.iscomplexobj(x)
        if noncomplex:
            x = x.astype(np.complex128)
        N = x.size
        x = np.r_[np.empty(self.order, x.dtype), x]
        y = np.empty(N+self.order, x.dtype)
        x_hist = self.x_hist
        y_hist = self.y_hist
        weave.inline(self.code, ['N','x_hist','y_hist','x','y'],
                     type_converters=weave.converters.blitz, verbose=0)
        if noncomplex:
            y = y.real
        return y[self.order:]
    def copy(self):
        other = self.__class__.__new__(self.__class__)
        other.code = self.code
        dtype = self.x_hist.dtype
        other.x_hist = np.zeros(self.order, dtype)
        other.y_hist = np.zeros(self.order, dtype)
        other.order = self.order
        return other

def MkFilter(method, type, order, alpha, ripple=None):
    if not isinstance(alpha, (tuple, list)):
        alpha = (alpha,)
    args = ['-%s' % method]
    if method == 'Ch':
        if ripple is None:
            ripple = defaultChebyshevRipple_dB
        args.append(ripple)
    args += ['-%s' % type, '-o', order, '-a'] + list(alpha)
    return IIRFilter(*mkfilter.mkfilter(*args))

def lowpass(freq, order=6, method='Bu', ripple=None):
    return MkFilter(method, 'Lp', order, freq, ripple)

def highpass(freq, order=6, method='Bu', ripple=None):
    return MkFilter(method, 'Hp', order, freq, ripple)

def bandpass(freq1, freq2, order=6, method='Bu', ripple=None):
    return MkFilter(method, 'Hp', order, (freq1, freq2), ripple)

def interpolate(sequence, factor):
    #return np.hstack((sequence[:,None] * factor,
    #                  np.zeros((sequence.size, factor-1), sequence.dtype))).flatten()
    return np.repeat(sequence, factor)

class stagedLowpass:
    def __init__(self, freq):
        self.freq = freq
        aa = lowpass(2**-9, method='Ch', ripple=-.021, order=4)
        upsampleSteps = max(0, int(-(np.log2(freq)+4)//6))
        self.down = [(aa.copy(), aa.copy()) for i in range(upsampleSteps)]
        self.final = lowpass(freq * 64**upsampleSteps)
        self.up = [(aa.copy(), aa.copy()) for i in range(upsampleSteps)]
        self.hist = np.zeros(0, np.complex128)
        self.quantum = 64**upsampleSteps
    def __call__(self, x):
        noncomplex = not np.iscomplexobj(x)
        if noncomplex:
            x = x.astype(np.complex128)
        x = np.r_[self.hist, x]
        n = x.size//self.quantum*self.quantum
        self.hist = x[n:]
        x = x[:n]
        for f1,f2 in self.down:
            x = f2(f1(x))[::64]
        x = self.final(x)
        for f1,f2 in self.up:
            x = f2(f1(interpolate(x, 64)))
        if noncomplex:
            x = x.real
        return x
    def copy(self):
        other = self.__class__.__new__(self.__class__)
        other.freq = self.freq
        other.down = [f.copy() for f in self.down]
        other.final = self.final.copy()
        other.up = [f.copy() for f in self.up]
        other.hist = self.hist.copy()
        other.quantum = self.quantum
        return other
