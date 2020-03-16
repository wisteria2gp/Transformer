
from torchsummary import summary
from tst.multiHeadAttention import MultiHeadAttention


if '__name__'=='__main__':
    x = (8, 100, 16)
    d_model = 16,
    q = 8,
    v = 8,
    h = 4,
    model = MultiHeadAttention(x[1], 8, 8, 4)
    print(model)
    print(summary(model, x))
    print("output")