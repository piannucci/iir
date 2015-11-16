lp = iir.lowpass(.45/upsample_factor, order=6)
y = lp((np.random.standard_normal(16384*256)+np.random.standard_normal(16384*256)*1j)*.5**.5).reshape(-1,1024)
window = np.blackman(y.shape[1])
window /= window.sum()**.5
pl.clf()
Syy = (np.abs(np.fft.fft(y*window))**2).mean(0)
pl.plot(np.arange(Syy.size,dtype=float)/Syy.size, 10*np.log10(Syy))
pl.vlines((np.arange(-52/2,52/2) + np.array([0,nfft,-nfft])[:,None])%(upsample_factor*nfft)/upsample_factor/nfft, 0, 20)

pl.figure(1)
pl.clf()
_=pl.specgram(output+np.random.standard_normal(output.shape)*1e-4,NFFT=nfft*upsample_factor,noverlap=63*upsample_factor,interpolation='nearest',Fs=1.)
pl.colorbar()
pl.xlim(0,output2.size)
pl.ylim(-2.5/upsample_factor,2.5/upsample_factor)

pl.figure(2)
pl.clf()
_=pl.specgram(lp(output)+np.random.standard_normal(output.shape)*1e-2,NFFT=nfft*upsample_factor,noverlap=63*upsample_factor,interpolation='nearest',Fs=1.)
pl.colorbar()
pl.xlim(0,output2.size)
pl.ylim(-2.5/upsample_factor,2.5/upsample_factor)

y0 = next(encodeBlurt([np.fromstring('Hello, world!', np.uint8)]))
pl.clf()
_=pl.specgram(y0, NFFT=64*32, noverlap=63*32, vmin=-50, vmax=0);pl.ylim(.35,.48)


# Y(z) = gamma * np.polyval(alpha[::-1], z) / (z**order - np.polyval(beta[::-1], z))

import iir, numpy as np, pylab as pl
upsample_factor = 32
nfft = 64
z=np.exp(1j*np.arange(0,10000)*np.pi*2/10000.)

pl.clf()
order, alpha, beta, gamma = iir.mkfilter.mkfilter('-Bu', '-Lp', '-o', 6, '-a', .45/32)
pl.plot(np.angle(z)/(2*np.pi), 40*np.log10(abs(gamma * np.polyval(alpha[::-1], z) / (z**order - np.polyval(beta[::-1], z)))))
order, alpha, beta, gamma = iir.mkfilter.mkfilter('-Ch', -.021, '-Lp', '-o', 6, '-a', 26.5/32/64)
pl.plot(np.angle(z)/(2*np.pi), 40*np.log10(abs(gamma * np.polyval(alpha[::-1], z) / (z**order - np.polyval(beta[::-1], z)))))
order, alpha, beta, gamma = iir.mkfilter.mkfilter('-Ch', -.05, '-Lp', '-o', 6, '-a', 26.5/32/64)
pl.plot(np.angle(z)/(2*np.pi), 40*np.log10(abs(gamma * np.polyval(alpha[::-1], z) / (z**order - np.polyval(beta[::-1], z)))))
pl.vlines((np.arange(-52/2,52/2) + np.array([0,nfft,-nfft])[:,None])/upsample_factor/nfft, -3, 3)
pl.xlim(-.5, .5)
pl.ylim(-100,100)
