
Pytorch  

# Tensor  

## Structure  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2F8q9p7TBLnG.png?alt=media&token=3816b3e2-a598-455d-aa8d-2aa136807472" width="500">


One thing to note about the dimension of a tensor is that it differs from what we mean when we refer to the dimension of a vector in a vector space. The dimension of a tensor does not tell us how many components exist within the tensor.  

If we have a three dimensional vector from three dimensional euclidean space, we have an ordered triple with three components.  

A three dimensional tensor, however, can have many more than three components. Our two dimensional tensor dd for example has nine components.  

> dd = [<br/>[1,2,3],<br/>[4,5,6],<br/>[7,8,9]<br/>]  

## Rank, Axes And Shape - Tensors For Deep Learning  

### Rank  

**The _rank_ of a tensor refers to the number of dimensions present within the tensor.** Suppose we are told that we have a rank-2 tensor. This means all of the following:  

We have a matrix  

We have a 2d-array  

We have a 2d-tensor  

### Rank And Indexes  

The rank of a tensor tells us how many indexes are required to access (refer to) a specific data element contained within the tensor data structure.  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FEkyXWotSDk.png?alt=media&token=376f778b-f2f1-4501-a8c6-08e91f574c64" width="400">  

### Axes  

An axis of a tensor is a specific dimension of a tensor.  

If we say that a tensor is a rank 2 tensor,   

we mean that the tensor has 2 dimensions,   

or equivalently, the tensor has two axes.  

### Length Of An Axis  

The length of each axis tells us how many indexes are available along each axis.  

Note that, with tensors, the elements of the last axis are always numbers. Every other axis will contain n-dimensional arrays. This is what we see in this example, but this idea generalizes.  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FytXXdp5MU1.png?alt=media&token=d0e02c62-0627-49f3-b6e3-bf8c98b989c8" width="400">  

### Shape Of A Tensor  

The _shape_ of a tensor is determined by the length of each axis, so if we know the shape of a given tensor, then we know the length of each axis, and this tells us how many indexes are available along each axis.  

The rank of a variable is equal to the length of its shape  

rank: `len(dd.shape)=2`  

Two rank tensor, or equivalently, the tensor has two axes.  

shape: `dd.shape`  

The length of each axes  

  

## Tensor computation  

Tensor computations depend on** the device and the type**. (VERSION 1.0)  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FS_wOHn5Kl8.png?alt=media&token=6d8a10b8-0b0d-489c-84d5-6337137bfe09" width="400">  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2F9gVaj-3-Y0.png?alt=media&token=c2af8358-b1dc-42d6-9a5c-46c5b51214f0" width="400">  

### PyTorch Tensor Type Promotion  

Arithmetic and comparison operations, as of PyTorch version 1.3, can perform mixed-type operations that promote to a common dtype.  

The example below was not allowed in version 1.2. However, in version 1.3 and above, the same code returns a tensor with dtype=torch.float32.  

torch.tensor([1], dtype=torch.int) + <br/>torch.tensor([1], dtype=torch.float32)  

See the [full documentation](https://github.com/pytorch/pytorch/blob/master/docs/source/tensor_attributes.rst#type-promotion-doc) for more details.  

torch.result_type Provide function to determine result of mixed-type ops [26012](https://github.com/pytorch/pytorch/pull/26012).  

torch.can_cast Expose casting rules for type promotion [26805](https://github.com/pytorch/pytorch/pull/26805).  

torch.promote_types Expose promotion logic [26655](https://github.com/pytorch/pytorch/pull/26655).  

## Shape of CNN input  

[Batch, Channels, Height, Width]  

Each image has a single color channel, and the image height and width are 28 x 28 respectively.  

- Batch size  

- Color channels  

- Height  

- Width  

This gives us a single rank-4 tensor that will ultimately flow through our convolutional neural network.  

Given a tensor of images like this, we can navigate to a specific pixel in a specific color channel of a specific image in the batch using four indexes.  

### NCHW vs NHWC vs CHWN  

It's common when reading API documentation and academic papers to see the B replaced by an N. The N standing for _number of samples_ in a batch.  

Furthermore, another difference we often encounter in the wild is a _reordering_ of the dimensions. Common orderings are as follows:  

NCHW  

NHWC  

CHWN  

As we have seen, PyTorch uses NCHW, and it is the case that TensorFlow and Keras use NHWC by default (it can be configured). Ultimately, the choice of which one to use depends mainly on performance. Some libraries and algorithms are more suited to one or the other of these orderings.  

## Feature map  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FdKZF2ULXpz.png?alt=media&token=b16e0bf9-a950-4cb5-b1b6-6c6d444926cb" width="400">  

# Pytorch Tensor  

### Instances Of The torch.Tensor Class  

PyTorch tensors are instances of the torch.Tensor Python class. We can create a torch.Tensor object using the class constructor like so:  

> t = torch.Tensor()<br/>> type(t)<br/>torch.Tensor  

This creates an empty tensor (tensor with no data), but we'll get to adding data in just a moment.  

### Tensor Attributes  

First, let's look at a few tensor attributes. Every torch.Tensor has these attributes:  

torch.dtype ▶ torch.float32  

**Tensors contain uniform (of the same type) numerical data with one of these types:**  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FK2olaDZo1C.png?alt=media&token=844499cd-a950-49bd-a3dd-2970ecb9abe8" width="400">

torch.device ▶ CPU  

torch.layout ▶ torch.strided  

The layout, strided in our case, specifies how the tensor is stored in memory. To learn more about stride check [here](https://en.wikipedia.org/wiki/Stride_of_an_array).  

### Creating Tensors Using Data  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FpONlVOukNf.png?alt=media&token=c5ba2cc7-dc74-4531-b114-3302d20fc141" width="400">  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FAfzTrIq1AV.png?alt=media&token=15c19c57-0aec-49c4-b169-b389c482cf9a" width="400">  

### Different ways to create tensor  

```python
data = np.array([1,2,3])
type(data)
> numpy.ndarray

# the constructor of the torch.Tensor class
o1 = torch.Tensor(data) 
> tensor([1., 2., 3.])

print(o1.dtype)
> torch.float32

# call a factory function that constructs torch.Tensor objects and returns them to the caller
torch.tensor(data) 
> tensor([1, 2, 3], dtype=torch.int32)

torch.as_tensor(data) 
> tensor([1, 2, 3], dtype=torch.int32)

torch.from_numpy(data)
> tensor([1, 2, 3], dtype=torch.int32)
```  

### Numpy dtype Behavior On Different Systems  

Depending on your machine and operating system, it is possible that your dtype may be different from what is shown here and in the video.  

Numpy sets its default dtype based on whether it's running on a 32-bit or 64-bit system, and the behavior also differs on Windows systems.  

This [link](https://stackoverflow.com/questions/36278590/numpy-array-dtype-is-coming-as-int32-by-default-in-a-windows-10-64-bit-machine) provides further information regrading the difference seen on Windows systems. The affected methods are: tensor, as_tensor, and from_numpy.  

### Default dtype Vs Inferred dtype  

the torch.Tensor() constructor uses the default dtype when building the tensor.  

```python
torch.get_default_dtype()
> torch.float32
```  

The other calls choose a dtype based on the incoming data. This is called **type inference**.  

With torch.Tensor(), we are unable to pass a dtype to the constructor. This is an example of the torch.Tensor() constructor lacking in configuration options. This is one of the reasons to go with the torch.tensor() factory function for creating our tensors.  

### Sharing Memory For Performance: Copy Vs Share  

However, after setting data[0]=0, we can see some of our tensors have changes. The first two o1 and o2 still have the original value of 1 for index 0, while the second two o3 and o4 have the new value of 0 for index 0.  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FkJOqtmgVYN.png?alt=media&token=44e3c1cf-4ba5-4560-9a3e-ba12ef34afac" width="400">  

This happens because torch.Tensor() and torch.tensor() _copy_ their input data while torch.as_tensor() and torch.from_numpy() _share_ their input data in memory with the original input object.  

**Sharing data is more efficient and uses less memory** than copying data because the data is not written to two locations in memory.  

  

- Share Data  
  - torch.as_tensor()  
  - torch.from_numpy() 

- Copy Data  
  - torch.tensor() 
  - torch.Tensor()  

### Best Options For Creating Tensors In PyTorch  

Given all of these details, these two are the best options:  

`torch.tensor()` : go-to call  

`torch.as_tensor()`: should be employed when tuning our code for performance.  

`torch.as_tensor()` is the winning choice in the memory sharing game.  

The torch.from_numpy() function only accepts numpy.ndarrays, while the torch.as_tensor() function accepts a wide variety of [array-like objects](https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays) including other PyTorch tensors.   

Some things to keep in mind about memory sharing (it works where it can):  

Since numpy.ndarray objects are allocated on the CPU, the as_tensor() function must copy the data from the CPU to the GPU when a GPU is being used.  

The memory sharing of as_tensor() doesn't work with built-in Python data structures like lists.  

The as_tensor() call requires developer knowledge of the sharing feature. This is necessary so we don't inadvertently make an unwanted change in the underlying data without realizing the change impacts multiple objects.  

The as_tensor() performance improvement will be greater if there are a lot of back and forth operations between numpy.ndarray objects and tensor objects. However, if there is just a single load operation, there shouldn't be much impact from a performance perspective.  

### Tensor Operation Types  

#### Reshaping operations  

```python
# get the shape
> t.size()
torch.Size([3, 4])

> t.shape
torch.Size([3, 4])

# get the rank
> len(t.shape)
2

# get the number of elements inside a tensor = the product of the shape's component values
> torch.tensor(t.shape).prod()
tensor(12)

> t.numel() # numel = number of elements
12
```  

squeeze and unsqueeze to change shape  

_Squeezing_ a tensor removes the dimensions or axes that have a length of one.  

_Unsqueezing_ a tensor adds a dimension with a length of one.  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FxOTk14f_MA.png?alt=media&token=e1ae2708-aeab-4dd3-9d40-82501a87ff51" width="400">


_Flattening_ a tensor means to remove all of the dimensions except for one.  

```python
def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
```  

Concatenating Tensors  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FE4j_MNnZZo.png?alt=media&token=e8436fd3-53c6-42d9-b81c-4dd62021cb80" width="450">

Stack Tensors  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FRPTAS137Kb.png?alt=media&token=fb02dfee-58ec-4481-bbde-2c34d18bf45b" width="400">  

#### Broadcasting and Element-wise(Component-wise/Point-wise) operations  

EX  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FgjyYntxZ4l.png?alt=media&token=80e48785-424a-4661-9f86-edab1e2544e2" width="400">

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FDemkHpady2.png?alt=media&token=a9f6b258-eeb4-4440-9ee1-cbf7338bbd00" width="400">  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FfxwNd8kJLc.png?alt=media&token=05bcfbf5-a98b-45cb-bdc1-2e6d4eee80e0" width="400">

**comparison operations**  

#### Reduction operations  

#### Access operations  

### CNN Flatten Operation Visualized  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FqlZIUgxkfj.png?alt=media&token=dd9f1193-d1be-48cc-a95a-a3b6baf4c0f5" width="400">  

smashes all the images together into a single axis which won't work well inside CNN  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FMsAqjT0Hdd.png?alt=media&token=580d8481-db67-497b-8c1a-a8181a19f6ec" width="400">

to flatten each image while still maintaining the batch axis. This means we want to flatten _only part of the tensor_. We want to flatten the, color channel axis with the height and width axes.  

<img src="https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FFemFarm%2FdXkHKdQEzh.png?alt=media&token=fba71853-4c44-4a96-936b-e6f2baaa78b1" width="400">  

```python
# built-in function
t.flatten(start_dim=1)

# another way to flatten
t.reshape(t.shape[0], -1)
```  
