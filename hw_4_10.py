import torch


class MyLSTMCell(torch.nn.Module):
    def __init__(self, wf, bf, wi, bi, wc, bc, wo, bo):
        super(MyLSTMCell, self).__init__()
        self._wf = wf
        self._bf = bf
        self._wi = wi
        self._bi = bi
        self._wc = wc
        self._bc = bc
        self._wo = wo
        self._bo = bo

    def forward(self, x: torch.Tensor, states_tuple: torch.Tensor):
        sigmoid = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()
        h, c = states_tuple
        input_concat = torch.cat((h, x))
        ft = sigmoid(torch.matmul(self._wf, input_concat) + self._bf)
        it = sigmoid(torch.matmul(self._wi, input_concat) + self._bi)
        ctd = tanh(torch.matmul(self._wc, input_concat) + self._bc)
        temp1 = ft * c
        temp2 = it * ctd
        cx = temp1 + temp2
        ot = sigmoid(torch.matmul(self._wo, input_concat) + self._bo)
        hx = ot * tanh(cx)
        return hx, cx


wf = torch.tensor([[1.1, 2.2, 3.3, 4.4, 1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 3.3, 4.4, 1.1, 2.2, 3.3, 4.4],
                   [1.1, 2.2, 3.3, 4.4, 1.1, 2.2, 3.3, 4.4], [1.1, 2.2, 3.3, 4.4, 1.1, 2.2, 3.3, 4.4]])
bf = torch.tensor([1.12, 2.22, 1.12, 2.22])
wi = torch.tensor([[1.13, 2.23, 3.33, 4.43, 1.13, 2.23, 3.33, 4.43], [1.13, 2.23, 3.33, 4.43, 1.13, 2.23, 3.33, 4.43],
                   [1.13, 2.23, 3.33, 4.43, 1.13, 2.23, 3.33, 4.43], [1.13, 2.23, 3.33, 4.43, 1.13, 2.23, 3.33, 4.43]])
bi = torch.tensor([1.14, 2.24, 1.14, 2.24])
wc = torch.tensor([[1.15, 2.25, 3.35, 4.45, 1.15, 2.25, 3.35, 4.45], [1.15, 2.25, 3.35, 4.45, 1.15, 2.25, 3.35, 4.45],
                   [1.15, 2.25, 3.35, 4.45, 1.15, 2.25, 3.35, 4.45], [1.15, 2.25, 3.35, 4.45, 1.15, 2.25, 3.35, 4.45]])
bc = torch.tensor([1.16, 2.26, 1.16, 2.26])
wo = torch.tensor([[1.17, 2.27, 3.37, 4.47, 1.17, 2.27, 3.37, 4.47], [1.17, 2.27, 3.37, 4.47, 1.17, 2.27, 3.37, 4.47],
                   [1.17, 2.27, 3.37, 4.47, 1.17, 2.27, 3.37, 4.47], [1.17, 2.27, 3.37, 4.47, 1.17, 2.27, 3.37, 4.47]])
bo = torch.tensor([1.18, 2.28, 1.18, 2.28])

cell = MyLSTMCell(wf, bf, wi, bi, wc, bc, wo, bo)
sample_x = torch.tensor([1., 2., 3., 4.])
sample_states_tuple = torch.tensor([[5., 6., 7., 8.], [5.5, 6.6, 7.7, 8.8]])
print(cell.forward(sample_x, sample_states_tuple))