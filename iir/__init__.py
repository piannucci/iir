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
# y[i] = gamma * sum(alpha[j] * x[i+j-order] for j in xrange(order+1)) +
#        sum(beta[j] * y[i+j-order] for j in xrange(order))
#
# and transfer function
# Y(z) = gamma * sum(alpha[j] * z**(j-order) for j in xrange(order+1)) / (1 - sum(beta[j] * z**(j-order) for j in xrange(order)))
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
        N = x.size
        x = np.r_[np.empty(self.order, x.dtype), x]
        y = np.empty(N+self.order, x.dtype)
        x_hist = self.x_hist
        y_hist = self.y_hist
        weave.inline(self.code, ['N','x_hist','y_hist','x','y'],
                     type_converters=weave.converters.blitz, verbose=0)
        return y[self.order:]
    def copy(self):
        other = IIRFilter.__new__(IIRFilter)
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
