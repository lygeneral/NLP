import tensorflow as tf
tf.__version__


# 张量
# 常量
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b)
print(x)
print(a + b)

a.get_shape()

a.numpy()

# 变量
s = tf.Variable(2, name="scalar")
m = tf.Variable([[0, 1], [2, 3]], name="matrix")
W = tf.Variable(tf.zeros([784, 10]))
s.assign(3)
s.assign_add(3)


class MyModule(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]


m = MyModule()
m.variables

# tf.data
dataset = tf.data.Dataset.from_tensors([1, 2, 3, 4, 5])
for element in dataset:
    print(element.numpy())
    it = iter(dataset)
print(next(it).numpy())

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
for element in dataset:
    print(element.numpy())
it = iter(dataset)
print(next(it).numpy())

features = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
labels = tf.data.Dataset.from_tensor_slices([6, 7, 8, 9, 10])
dataset = tf.data.Dataset.zip((features, labels))
for element in dataset:
    print(element)

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
    print([arr.numpy() for arr in batch])

shuffle_dataset = dataset.shuffle(buffer_size=10)
for element in shuffle_dataset:
    print(element)

shuffle_dataset = dataset.shuffle(buffer_size=100)
for element in shuffle_dataset:
    print(element)
