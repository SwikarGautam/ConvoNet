import numpy as np


class Maxpool:

    def __init__(self, window_shape, stride=None, pad=0):
        self.input_shape = None
        self.window_shape = window_shape
        self.stride = stride if stride else window_shape
        self.pad = pad
        self.output_shape = 0
        self.output = None
        self.training = False
        self.input = None
        self.extra = None
        self.n = None
        self.d_list = []

    def forward(self, input_layer):
        self.input = input_layer
        self.input_shape = input_layer.shape
        n0 = (input_layer.shape[0] - self.window_shape + 2 * self.pad) // self.stride + 1
        n1 = (input_layer.shape[1] - self.window_shape + 2 * self.pad) // self.stride + 1
        self.output_shape = (n0, n1, input_layer.shape[2], input_layer.shape[3])
        if self.stride == self.window_shape:
            out = input_layer[:int((input_layer.shape[0] // self.window_shape) * self.window_shape),
                              :int((input_layer.shape[1] // self.window_shape) * self.window_shape)]
            out = np.maximum.reduceat(out, np.arange(0, out.shape[0], self.stride), axis=0)
            out = np.maximum.reduceat(out, np.arange(0, out.shape[1], self.stride), axis=1)
            self.output = out
            return self.output

        else:
            for i in range(1, self.window_shape + 1):
                if i * self.stride >= self.window_shape:
                    n = i
                    extra = (i * self.stride) - self.window_shape
                    break

            def func(s, y):
                x = np.arange(s, y)
                odd = np.remainder(x, 2).astype(bool)
                even = np.invert(odd)
                x[odd] = self.window_shape + (x[odd] // 2) * (self.window_shape + extra)
                x[even] = (x[even] // 2) * (self.window_shape + extra)
                x = x[x < y]
                return x

            t = self.output_shape
            out = np.zeros((t[0], t[1], input_layer.shape[2], t[3]))
            for c1, i in enumerate(range(0, self.window_shape, self.stride)):

                for c2, j in enumerate(range(0, self.window_shape, self.stride)):
                    temp = input_layer[i:, j:]
                    a = int(((temp.shape[0] // (self.window_shape + extra)) * (self.window_shape + extra)) +
                            (self.window_shape + extra if temp.shape[0] % (self.window_shape + extra) ==
                            self.window_shape else 0))
                    b = int(((temp.shape[1] // (self.window_shape + extra)) * (self.window_shape + extra)) +
                            (self.window_shape + extra if temp.shape[1] % (self.window_shape + extra) ==
                            self.window_shape else 0))
                    temp = temp[:a, :b]

                    self.d_list.append((a, b))

                    temp = np.maximum.reduceat(temp, func(0, temp.shape[0]).tolist(), axis=0)[::2]

                    temp = np.maximum.reduceat(temp, func(0, temp.shape[1]).tolist(), axis=1)[:, ::2]

                    out[c1::n, c2::n] = temp

            self.extra = extra
            self.n = n
            self.output = out
            return self.output

    def find_gradient(self, delta):

        assert delta.shape[:2] == self.output_shape[:2]
        if self.stride == self.window_shape:
            delta = np.repeat(np.repeat(delta, self.stride, axis=1), self.stride, axis=0)
            out = np.repeat(np.repeat(self.output, self.stride, axis=1), self.stride, axis=0)
            input1 = self.input[:out.shape[0], :out.shape[1]]
            out_delta = np.where(np.equal(out, input1), delta, 0)
            temp = np.zeros_like(self.input)
            temp[:out_delta.shape[0], :out_delta.shape[1]] = out_delta
            return temp

        else:
            out_delta = np.zeros_like(self.input)
            s = 0
            mask_t = np.zeros((self.window_shape + self.extra))
            mask_t[:self.window_shape] = 1
            mask_r = np.tile(mask_t, (self.input_shape[0] // mask_t.size + 1))
            mask_c = np.tile(mask_t, (self.input_shape[1] // mask_t.size + 1))
            mask = np.outer(mask_r, mask_c).astype(bool)
            _, _, i2, i3 = self.input_shape
            mask = np.tile(np.expand_dims(np.expand_dims(mask, axis=2), axis=3), (1, 1, i2, i3))
            for c1, i in enumerate(range(0, self.window_shape, self.stride)):
                for c2, j in enumerate(range(0, self.window_shape, self.stride)):
                    temp2 = self.output[c1::self.n, c2::self.n]
                    temp3 = delta[c1::self.n, c2::self.n]
                    a, b = self.d_list[s]
                    temp = self.input[i:i + a, j:j + b]
                    temp2 = np.repeat(np.repeat(temp2, self.window_shape + self.extra, axis=1),
                                      self.window_shape + self.extra, axis=0)
                    temp3 = np.repeat(np.repeat(temp3, self.window_shape + self.extra, axis=1),
                                      self.window_shape + self.extra, axis=0)
                    if temp2.shape != temp.shape:
                        temp2 = temp2[:temp.shape[0], :temp.shape[1]]
                        temp3 = temp3[:temp.shape[0], :temp.shape[1]]
                    mask1 = mask[:a, :b]
                    out_delta[i:i + a, j:j + b] += np.where(np.bitwise_and(np.equal(temp, temp2), mask1), temp3, 0)
                    s += 1
            return out_delta

    def update(self, delta, learning_rate, mini_size):
        delta_z = self.find_gradient(delta)
        delta_z = delta_z[self.pad:self.pad + self.input_shape[0], self.pad:self.pad + self.input_shape[1]]
        return delta_z
