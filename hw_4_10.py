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
        ctd = tanh(torch.matmul(self._wc, input_concat), self._bc)
        cx = ft * c + it * ctd
        ot = sigmoid(torch.matmul(self._wo, input_concat) + self._bo)
        hx = ot * tanh(cx)
        return hx, cx

