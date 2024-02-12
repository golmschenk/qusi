import torch

from qusi.hadryss_model import Hadryss


def test_lengths_give_correct_output_size():
    hadryss50 = Hadryss(input_length=50)

    output50 = hadryss50(torch.arange(50, dtype=torch.float32).reshape([1, 50]))

    assert output50.shape == torch.Size([1])

    hadryss1000 = Hadryss(input_length=1000)

    output1000 = hadryss1000(torch.arange(1000, dtype=torch.float32).reshape([1, 1000]))

    assert output1000.shape == torch.Size([1])

    hadryss3673 = Hadryss(input_length=3673)

    output3673 = hadryss3673(torch.arange(3673, dtype=torch.float32).reshape([1, 3673]))

    assert output3673.shape == torch.Size([1])

    hadryss100000 = Hadryss(input_length=100000)

    output100000 = hadryss100000(torch.arange(100000, dtype=torch.float32).reshape([1, 100000]))

    assert output100000.shape == torch.Size([1])
