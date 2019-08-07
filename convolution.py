import numpy as np


class Conv:

    def __init__(self, input_shape, weights_shape, stride=1, padding=0):
        self.input_shape = input_shape
        self.weights_shape = weights_shape
        self.weights = np.random.randn(*weights_shape)
        self.biases = np.random.randn(1, 1, weights_shape[2])
        self.stride = stride
        self.pad = padding
        n0 = (self.input_shape[0] - self.weights_shape[0] + 2 * padding) // self.stride + 1
        n1 = (self.input_shape[1] - self.weights_shape[1] + 2 * padding) // self.stride + 1
        self.output_shape = n0, n1, input_shape[2], weights_shape[2]
        self.output = np.zeros(self.output_shape)
        self.input = None
        self.vw = 0
        self.vb = 0
        self.sw = 0
        self.sb = 0

    def out(self, input_layer, weights=None, stride=None, pad=None, d_conv=False, d_a=False, delta=None):
        if not d_conv:
            if not d_a:
                self.input = input_layer
            weights = self.weights
            stride = self.stride
            pad = self.pad
        weights = weights.copy()
        input_layer = input_layer.copy()
        i0, i1, i2, i3 = input_layer.shape
        if pad:
            temp_layer = np.zeros((i0 + 2 * pad, i1 + 2 * pad, i2, i3))
            temp_layer[pad:pad + i0, pad:i1 + pad] = input_layer
            input_layer = temp_layer
        i0, i1, _, _ = input_layer.shape
        w0, w1, w2, w3 = weights.shape
        n00 = (i0 - w0) // stride + 1
        n01 = (i1 - w1) // stride + 1
        if d_conv:
            input_layer = np.expand_dims(input_layer, axis=3)
            weights = np.expand_dims(weights, axis=4)
            out_shape = (n00, n01, i2, w3, i3)
        elif d_a:
            delta = np.expand_dims(delta.copy(), axis=4)
            weights = np.expand_dims(weights, axis=2)
            out_shape = 1, 1
        else:
            input_layer = np.expand_dims(input_layer, axis=3)
            weights = np.expand_dims(weights, axis=2)
            out_shape = (n00, n01, i2, w2, w3)
        _, _, w2, w3, w4 = weights.shape
        out = np.zeros(out_shape)
        for i in range(1, w0 + 1):
            if i * stride >= w0:
                n1 = i
                extra1 = (i * stride) - w0
                break
            if i + w0 > i0:
                n1 = i
                extra1 = 0
                break
        n2 = n1
        extra2 = extra1
        if w0 != w1:
            for i in range(1, w1 + 1):
                if i * stride >= w1:
                    n2 = i
                    extra2 = (i * stride) - w1
                    break
                if i + w1 > i1:
                    n2 = i
                    extra2 = 0
                    break
        d = np.zeros((w0 + extra1, w1 + extra2, w2, w3, w4))

        if extra1 or extra2:
            d[:w0, :w1] = weights
            weights = d
        w0, w1, _, _, _ = weights.shape

        temp1 = i0 // w0 if (i0 % w0) < (w0 - extra1) else i0 // w0 + 1
        temp2 = i1 // w1 if (i1 % w1) < (w1 - extra2) else i1 // w1 + 1
        fil = np.tile(weights, (temp1, temp2, 1, 1, 1))

        def func1(x):
            odd = np.remainder(x, 2).astype(bool)
            even = np.invert(odd)
            x[odd] = w0 - extra1 + (x[odd] // 2) * w0
            x[even] = (x[even] // 2) * w0
            x = x[x < x.size]
            return x

        def func2(x):
            odd = np.remainder(x, 2).astype(bool)
            even = np.invert(odd)
            x[odd] = w1 - extra2 + (x[odd] // 2) * w1
            x[even] = (x[even] // 2) * w1
            x = x[x < x.size]
            return x

        for c1, j in enumerate(range(0, w0, stride)):
            for c2, i in enumerate(range(0, w1, stride)):
                s1 = (j + fil.shape[0]) - i0
                s2 = (i + fil.shape[1]) - i1
                temp3 = fil
                if s1 > 0:
                    if s1 > extra1:
                        temp3 = temp3[:-w0]
                    else:
                        temp3 = temp3[:-s1]
                if s2 > 0:
                    if s2 > extra2:
                        temp3 = temp3[:, : -w1]
                    else:
                        temp3 = temp3[:, :-s2]

                if d_a:
                    delt = np.repeat(np.repeat(delta[c1::n1, c2::n2], w0, 0), w1, 1) \
                               [:temp3.shape[0], :temp3.shape[1]] * temp3
                    input_layer[j:j + temp3.shape[0], i:i + temp3.shape[1]] += delt.sum(axis=3)
                else:
                    b = temp3 * input_layer[j:j + temp3.shape[0], i:i + temp3.shape[1]]
                    b = np.add.reduceat(b, np.fromfunction(func1, (b.shape[0],)).tolist(), axis=0)[::2]
                    b = np.add.reduceat(b, np.fromfunction(func2, (b.shape[1],)).tolist(), axis=1)[:, ::2]
                    out[c1::n1, c2::n2] = b
        if d_a:
            return input_layer
        if d_conv:
            return out.sum(axis=2)
        self.output = out.sum(axis=4)  
        return self.output

    def find_gradient(self, delta):
        s1, s2, s3, s4 = delta.shape
        delta_b = np.add.reduce(delta.sum(axis=2), axis=(0, 1), keepdims=True)
        delta_a = self.out(np.zeros_like(self.input), d_a=True, delta=delta)
        expanded = np.zeros((s1 * self.stride - (self.stride - 1), s2 * self.stride - (self.stride - 1), s3, s4))
        expanded[::self.stride, ::self.stride] = delta
        temp = self.out(self.input, expanded, 1, self.pad, True)
        delta_a = delta_a[self.pad:self.pad + self.input_shape[0],
                  self.pad:self.pad + self.input_shape[1]]
        delta = temp[:self.weights_shape[0], :self.weights_shape[1]]
        return delta, delta_b, delta_a

    def update(self, delta, learning_rate, beta1=0.9, beta2=0.999):
        delta_w, delta_b, delta_z = self.find_gradient(delta)
        self.vw = beta1 * self.vw + (1 - beta1) * delta_w
        self.vb = beta1 * self.vb + (1 - beta1) * delta_b
        self.sw = beta2 * self.sw + (1 - beta2) * np.square(delta_w)
        self.sb = beta2 * self.sb + (1 - beta2) * np.square(delta_b)

        self.weights -= learning_rate * self.vw / np.sqrt(self.sw + 1e-8)
        return delta_z
