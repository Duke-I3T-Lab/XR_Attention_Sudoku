Ф°
ъ#╬#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
?
FloorMod
x"T
y"T
z"T"
Ttype:
2	
·
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%╖╤8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
e
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2		
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
л
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type*
output_handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements#
handleКщelement_dtype"
element_dtypetype"

shape_typetype:
2	
И
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
Ф
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
И"serve*2.6.02unknown8ф│
░
&digit_caps/digit_caps_transform_tensorVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&digit_caps/digit_caps_transform_tensor
й
:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOpReadVariableOp&digit_caps/digit_caps_transform_tensor*&
_output_shapes
:
*
dtype0
а
 digit_caps/digit_caps_log_priorsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" digit_caps/digit_caps_log_priors
Щ
4digit_caps/digit_caps_log_priors/Read/ReadVariableOpReadVariableOp digit_caps/digit_caps_log_priors*"
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
о
%feature_maps/feature_map_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/kernel
з
9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/kernel*&
_output_shapes
: *
dtype0
Ю
#feature_maps/feature_map_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_conv1/bias
Ч
7feature_maps/feature_map_conv1/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv1/bias*
_output_shapes
: *
dtype0
а
$feature_maps/feature_map_norm1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$feature_maps/feature_map_norm1/gamma
Щ
8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm1/gamma*
_output_shapes
: *
dtype0
Ю
#feature_maps/feature_map_norm1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#feature_maps/feature_map_norm1/beta
Ч
7feature_maps/feature_map_norm1/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm1/beta*
_output_shapes
: *
dtype0
о
%feature_maps/feature_map_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%feature_maps/feature_map_conv2/kernel
з
9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/kernel*&
_output_shapes
: @*
dtype0
Ю
#feature_maps/feature_map_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv2/bias
Ч
7feature_maps/feature_map_conv2/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv2/bias*
_output_shapes
:@*
dtype0
а
$feature_maps/feature_map_norm2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm2/gamma
Щ
8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm2/gamma*
_output_shapes
:@*
dtype0
Ю
#feature_maps/feature_map_norm2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm2/beta
Ч
7feature_maps/feature_map_norm2/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm2/beta*
_output_shapes
:@*
dtype0
о
%feature_maps/feature_map_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%feature_maps/feature_map_conv3/kernel
з
9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/kernel*&
_output_shapes
:@@*
dtype0
Ю
#feature_maps/feature_map_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_conv3/bias
Ч
7feature_maps/feature_map_conv3/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv3/bias*
_output_shapes
:@*
dtype0
а
$feature_maps/feature_map_norm3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*5
shared_name&$feature_maps/feature_map_norm3/gamma
Щ
8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm3/gamma*
_output_shapes
:@*
dtype0
Ю
#feature_maps/feature_map_norm3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#feature_maps/feature_map_norm3/beta
Ч
7feature_maps/feature_map_norm3/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm3/beta*
_output_shapes
:@*
dtype0
п
%feature_maps/feature_map_conv4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*6
shared_name'%feature_maps/feature_map_conv4/kernel
и
9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/kernel*'
_output_shapes
:@А*
dtype0
Я
#feature_maps/feature_map_conv4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#feature_maps/feature_map_conv4/bias
Ш
7feature_maps/feature_map_conv4/bias/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_conv4/bias*
_output_shapes	
:А*
dtype0
б
$feature_maps/feature_map_norm4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*5
shared_name&$feature_maps/feature_map_norm4/gamma
Ъ
8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOpReadVariableOp$feature_maps/feature_map_norm4/gamma*
_output_shapes	
:А*
dtype0
Я
#feature_maps/feature_map_norm4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#feature_maps/feature_map_norm4/beta
Ш
7feature_maps/feature_map_norm4/beta/Read/ReadVariableOpReadVariableOp#feature_maps/feature_map_norm4/beta*
_output_shapes	
:А*
dtype0
п
%primary_caps/primary_cap_dconv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*6
shared_name'%primary_caps/primary_cap_dconv/kernel
и
9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/kernel*'
_output_shapes
:		А*
dtype0
Я
#primary_caps/primary_cap_dconv/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*4
shared_name%#primary_caps/primary_cap_dconv/bias
Ш
7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOpReadVariableOp#primary_caps/primary_cap_dconv/bias*
_output_shapes	
:А*
dtype0
м
*feature_maps/feature_map_norm1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*feature_maps/feature_map_norm1/moving_mean
е
>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm1/moving_mean*
_output_shapes
: *
dtype0
┤
.feature_maps/feature_map_norm1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.feature_maps/feature_map_norm1/moving_variance
н
Bfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm1/moving_variance*
_output_shapes
: *
dtype0
м
*feature_maps/feature_map_norm2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm2/moving_mean
е
>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm2/moving_mean*
_output_shapes
:@*
dtype0
┤
.feature_maps/feature_map_norm2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm2/moving_variance
н
Bfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm2/moving_variance*
_output_shapes
:@*
dtype0
м
*feature_maps/feature_map_norm3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*feature_maps/feature_map_norm3/moving_mean
е
>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm3/moving_mean*
_output_shapes
:@*
dtype0
┤
.feature_maps/feature_map_norm3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.feature_maps/feature_map_norm3/moving_variance
н
Bfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm3/moving_variance*
_output_shapes
:@*
dtype0
н
*feature_maps/feature_map_norm4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*;
shared_name,*feature_maps/feature_map_norm4/moving_mean
ж
>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpReadVariableOp*feature_maps/feature_map_norm4/moving_mean*
_output_shapes	
:А*
dtype0
╡
.feature_maps/feature_map_norm4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*?
shared_name0.feature_maps/feature_map_norm4/moving_variance
о
Bfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOpReadVariableOp.feature_maps/feature_map_norm4/moving_variance*
_output_shapes	
:А*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
┤
(digit_caps/digit_caps_transform_tensor/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(digit_caps/digit_caps_transform_tensor/m
н
<digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOpReadVariableOp(digit_caps/digit_caps_transform_tensor/m*&
_output_shapes
:
*
dtype0
д
"digit_caps/digit_caps_log_priors/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"digit_caps/digit_caps_log_priors/m
Э
6digit_caps/digit_caps_log_priors/m/Read/ReadVariableOpReadVariableOp"digit_caps/digit_caps_log_priors/m*"
_output_shapes
:
*
dtype0
▓
'feature_maps/feature_map_conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'feature_maps/feature_map_conv1/kernel/m
л
;feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv1/kernel/m*&
_output_shapes
: *
dtype0
в
%feature_maps/feature_map_conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/bias/m
Ы
9feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/bias/m*
_output_shapes
: *
dtype0
д
&feature_maps/feature_map_norm1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&feature_maps/feature_map_norm1/gamma/m
Э
:feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm1/gamma/m*
_output_shapes
: *
dtype0
в
%feature_maps/feature_map_norm1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_norm1/beta/m
Ы
9feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm1/beta/m*
_output_shapes
: *
dtype0
▓
'feature_maps/feature_map_conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'feature_maps/feature_map_conv2/kernel/m
л
;feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv2/kernel/m*&
_output_shapes
: @*
dtype0
в
%feature_maps/feature_map_conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv2/bias/m
Ы
9feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/bias/m*
_output_shapes
:@*
dtype0
д
&feature_maps/feature_map_norm2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm2/gamma/m
Э
:feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm2/gamma/m*
_output_shapes
:@*
dtype0
в
%feature_maps/feature_map_norm2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm2/beta/m
Ы
9feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm2/beta/m*
_output_shapes
:@*
dtype0
▓
'feature_maps/feature_map_conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'feature_maps/feature_map_conv3/kernel/m
л
;feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv3/kernel/m*&
_output_shapes
:@@*
dtype0
в
%feature_maps/feature_map_conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv3/bias/m
Ы
9feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/bias/m*
_output_shapes
:@*
dtype0
д
&feature_maps/feature_map_norm3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm3/gamma/m
Э
:feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm3/gamma/m*
_output_shapes
:@*
dtype0
в
%feature_maps/feature_map_norm3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm3/beta/m
Ы
9feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm3/beta/m*
_output_shapes
:@*
dtype0
│
'feature_maps/feature_map_conv4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*8
shared_name)'feature_maps/feature_map_conv4/kernel/m
м
;feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv4/kernel/m*'
_output_shapes
:@А*
dtype0
г
%feature_maps/feature_map_conv4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%feature_maps/feature_map_conv4/bias/m
Ь
9feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/bias/m*
_output_shapes	
:А*
dtype0
е
&feature_maps/feature_map_norm4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&feature_maps/feature_map_norm4/gamma/m
Ю
:feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm4/gamma/m*
_output_shapes	
:А*
dtype0
г
%feature_maps/feature_map_norm4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%feature_maps/feature_map_norm4/beta/m
Ь
9feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm4/beta/m*
_output_shapes	
:А*
dtype0
│
'primary_caps/primary_cap_dconv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*8
shared_name)'primary_caps/primary_cap_dconv/kernel/m
м
;primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOpReadVariableOp'primary_caps/primary_cap_dconv/kernel/m*'
_output_shapes
:		А*
dtype0
г
%primary_caps/primary_cap_dconv/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%primary_caps/primary_cap_dconv/bias/m
Ь
9primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/bias/m*
_output_shapes	
:А*
dtype0
┤
(digit_caps/digit_caps_transform_tensor/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(digit_caps/digit_caps_transform_tensor/v
н
<digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOpReadVariableOp(digit_caps/digit_caps_transform_tensor/v*&
_output_shapes
:
*
dtype0
д
"digit_caps/digit_caps_log_priors/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"digit_caps/digit_caps_log_priors/v
Э
6digit_caps/digit_caps_log_priors/v/Read/ReadVariableOpReadVariableOp"digit_caps/digit_caps_log_priors/v*"
_output_shapes
:
*
dtype0
▓
'feature_maps/feature_map_conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'feature_maps/feature_map_conv1/kernel/v
л
;feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv1/kernel/v*&
_output_shapes
: *
dtype0
в
%feature_maps/feature_map_conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_conv1/bias/v
Ы
9feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv1/bias/v*
_output_shapes
: *
dtype0
д
&feature_maps/feature_map_norm1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&feature_maps/feature_map_norm1/gamma/v
Э
:feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm1/gamma/v*
_output_shapes
: *
dtype0
в
%feature_maps/feature_map_norm1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%feature_maps/feature_map_norm1/beta/v
Ы
9feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm1/beta/v*
_output_shapes
: *
dtype0
▓
'feature_maps/feature_map_conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*8
shared_name)'feature_maps/feature_map_conv2/kernel/v
л
;feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv2/kernel/v*&
_output_shapes
: @*
dtype0
в
%feature_maps/feature_map_conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv2/bias/v
Ы
9feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv2/bias/v*
_output_shapes
:@*
dtype0
д
&feature_maps/feature_map_norm2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm2/gamma/v
Э
:feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm2/gamma/v*
_output_shapes
:@*
dtype0
в
%feature_maps/feature_map_norm2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm2/beta/v
Ы
9feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm2/beta/v*
_output_shapes
:@*
dtype0
▓
'feature_maps/feature_map_conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*8
shared_name)'feature_maps/feature_map_conv3/kernel/v
л
;feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv3/kernel/v*&
_output_shapes
:@@*
dtype0
в
%feature_maps/feature_map_conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_conv3/bias/v
Ы
9feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv3/bias/v*
_output_shapes
:@*
dtype0
д
&feature_maps/feature_map_norm3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&feature_maps/feature_map_norm3/gamma/v
Э
:feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm3/gamma/v*
_output_shapes
:@*
dtype0
в
%feature_maps/feature_map_norm3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%feature_maps/feature_map_norm3/beta/v
Ы
9feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm3/beta/v*
_output_shapes
:@*
dtype0
│
'feature_maps/feature_map_conv4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*8
shared_name)'feature_maps/feature_map_conv4/kernel/v
м
;feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOpReadVariableOp'feature_maps/feature_map_conv4/kernel/v*'
_output_shapes
:@А*
dtype0
г
%feature_maps/feature_map_conv4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%feature_maps/feature_map_conv4/bias/v
Ь
9feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_conv4/bias/v*
_output_shapes	
:А*
dtype0
е
&feature_maps/feature_map_norm4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&feature_maps/feature_map_norm4/gamma/v
Ю
:feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOpReadVariableOp&feature_maps/feature_map_norm4/gamma/v*
_output_shapes	
:А*
dtype0
г
%feature_maps/feature_map_norm4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%feature_maps/feature_map_norm4/beta/v
Ь
9feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOpReadVariableOp%feature_maps/feature_map_norm4/beta/v*
_output_shapes	
:А*
dtype0
│
'primary_caps/primary_cap_dconv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		А*8
shared_name)'primary_caps/primary_cap_dconv/kernel/v
м
;primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOpReadVariableOp'primary_caps/primary_cap_dconv/kernel/v*'
_output_shapes
:		А*
dtype0
г
%primary_caps/primary_cap_dconv/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%primary_caps/primary_cap_dconv/bias/v
Ь
9primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpReadVariableOp%primary_caps/primary_cap_dconv/bias/v*
_output_shapes	
:А*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *є╡>

NoOpNoOp
иЖ
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*рЕ
value╒ЕB╤Е B╔Е
А
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures

_init_input_shape
к
	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
trainable_variables
	variables
regularization_losses
	keras_api
v
	dconv
reshape

squash
trainable_variables
	variables
regularization_losses
	keras_api
и
 digit_caps_transform_tensor
 W
!digit_caps_log_priors
!B

"squash
#trainable_variables
$	variables
%regularization_losses
&	keras_api
R
'trainable_variables
(	variables
)regularization_losses
*	keras_api
╨
+iter

,beta_1

-beta_2
	.decay
/learning_rate m▐!m▀0mр1mс2mт3mу4mф5mх6mц7mч8mш9mщ:mъ;mы<mь=mэ>mю?mя@mЁAmё vЄ!vє0vЇ1vї2vЎ3vў4v°5v∙6v·7v√8v№9v¤:v■;v <vА=vБ>vВ?vГ@vДAvЕ
Ц
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
 18
!19
╓
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
B16
C17
D18
E19
F20
G21
H22
I23
@24
A25
 26
!27
 
н
Jmetrics
trainable_variables

Klayers
Llayer_regularization_losses
Mnon_trainable_variables
	variables
Nlayer_metrics
	regularization_losses
 
 
h

0kernel
1bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
Ч
Saxis
	2gamma
3beta
Bmoving_mean
Cmoving_variance
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
h

4kernel
5bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
Ч
\axis
	6gamma
7beta
Dmoving_mean
Emoving_variance
]trainable_variables
^	variables
_regularization_losses
`	keras_api
h

8kernel
9bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
Ч
eaxis
	:gamma
;beta
Fmoving_mean
Gmoving_variance
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

<kernel
=bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
Ч
naxis
	>gamma
?beta
Hmoving_mean
Imoving_variance
otrainable_variables
p	variables
qregularization_losses
r	keras_api
v
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
╢
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
B16
C17
D18
E19
F20
G21
H22
I23
 
н
smetrics
trainable_variables

tlayers
ulayer_regularization_losses
vnon_trainable_variables
	variables
wlayer_metrics
regularization_losses
h

@kernel
Abias
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
R
|trainable_variables
}	variables
~regularization_losses
	keras_api
V
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api

@0
A1

@0
A1
 
▓
Дmetrics
trainable_variables
Еlayers
 Жlayer_regularization_losses
Зnon_trainable_variables
	variables
Иlayer_metrics
regularization_losses
ИЕ
VARIABLE_VALUE&digit_caps/digit_caps_transform_tensorKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE digit_caps/digit_caps_log_priorsElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUE
V
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api

 0
!1

 0
!1
 
▓
Нmetrics
#trainable_variables
Оlayers
 Пlayer_regularization_losses
Рnon_trainable_variables
$	variables
Сlayer_metrics
%regularization_losses
 
 
 
▓
Тmetrics
'trainable_variables
Уlayers
 Фlayer_regularization_losses
Хnon_trainable_variables
(	variables
Цlayer_metrics
)regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%feature_maps/feature_map_conv1/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#feature_maps/feature_map_conv1/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$feature_maps/feature_map_norm1/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#feature_maps/feature_map_norm1/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%feature_maps/feature_map_conv2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#feature_maps/feature_map_conv2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$feature_maps/feature_map_norm2/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#feature_maps/feature_map_norm2/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%feature_maps/feature_map_conv3/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#feature_maps/feature_map_conv3/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE$feature_maps/feature_map_norm3/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feature_maps/feature_map_norm3/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%feature_maps/feature_map_conv4/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feature_maps/feature_map_conv4/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE$feature_maps/feature_map_norm4/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#feature_maps/feature_map_norm4/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE%primary_caps/primary_cap_dconv/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#primary_caps/primary_cap_dconv/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*feature_maps/feature_map_norm1/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.feature_maps/feature_map_norm1/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*feature_maps/feature_map_norm2/moving_mean'variables/18/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.feature_maps/feature_map_norm2/moving_variance'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*feature_maps/feature_map_norm3/moving_mean'variables/20/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.feature_maps/feature_map_norm3/moving_variance'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE*feature_maps/feature_map_norm4/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.feature_maps/feature_map_norm4/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE

Ч0
Ш1
#
0
1
2
3
4
 
8
B0
C1
D2
E3
F4
G5
H6
I7
 

00
11

00
11
 
▓
Щmetrics
Otrainable_variables
Ъlayers
 Ыlayer_regularization_losses
Ьnon_trainable_variables
P	variables
Эlayer_metrics
Qregularization_losses
 

20
31

20
31
B2
C3
 
▓
Юmetrics
Ttrainable_variables
Яlayers
 аlayer_regularization_losses
бnon_trainable_variables
U	variables
вlayer_metrics
Vregularization_losses

40
51

40
51
 
▓
гmetrics
Xtrainable_variables
дlayers
 еlayer_regularization_losses
жnon_trainable_variables
Y	variables
зlayer_metrics
Zregularization_losses
 

60
71

60
71
D2
E3
 
▓
иmetrics
]trainable_variables
йlayers
 кlayer_regularization_losses
лnon_trainable_variables
^	variables
мlayer_metrics
_regularization_losses

80
91

80
91
 
▓
нmetrics
atrainable_variables
оlayers
 пlayer_regularization_losses
░non_trainable_variables
b	variables
▒layer_metrics
cregularization_losses
 

:0
;1

:0
;1
F2
G3
 
▓
▓metrics
ftrainable_variables
│layers
 ┤layer_regularization_losses
╡non_trainable_variables
g	variables
╢layer_metrics
hregularization_losses

<0
=1

<0
=1
 
▓
╖metrics
jtrainable_variables
╕layers
 ╣layer_regularization_losses
║non_trainable_variables
k	variables
╗layer_metrics
lregularization_losses
 

>0
?1

>0
?1
H2
I3
 
▓
╝metrics
otrainable_variables
╜layers
 ╛layer_regularization_losses
┐non_trainable_variables
p	variables
└layer_metrics
qregularization_losses
 
8
0
1
2
3
4
5
6
7
 
8
B0
C1
D2
E3
F4
G5
H6
I7
 

@0
A1

@0
A1
 
▓
┴metrics
xtrainable_variables
┬layers
 ├layer_regularization_losses
─non_trainable_variables
y	variables
┼layer_metrics
zregularization_losses
 
 
 
▓
╞metrics
|trainable_variables
╟layers
 ╚layer_regularization_losses
╔non_trainable_variables
}	variables
╩layer_metrics
~regularization_losses
 
 
 
╡
╦metrics
Аtrainable_variables
╠layers
 ═layer_regularization_losses
╬non_trainable_variables
Б	variables
╧layer_metrics
Вregularization_losses
 

0
1
2
 
 
 
 
 
 
╡
╨metrics
Йtrainable_variables
╤layers
 ╥layer_regularization_losses
╙non_trainable_variables
К	variables
╘layer_metrics
Лregularization_losses
 

"0
 
 
 
 
 
 
 
 
8

╒total

╓count
╫	variables
╪	keras_api
I

┘total

┌count
█
_fn_kwargs
▄	variables
▌	keras_api
 
 
 
 
 
 
 
 

B0
C1
 
 
 
 
 
 
 
 
 

D0
E1
 
 
 
 
 
 
 
 
 

F0
G1
 
 
 
 
 
 
 
 
 

H0
I1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

╒0
╓1

╫	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

┘0
┌1

▄	variables
жг
VARIABLE_VALUE(digit_caps/digit_caps_transform_tensor/mglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE"digit_caps/digit_caps_log_priors/malayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv1/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv1/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE&feature_maps/feature_map_norm1/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_norm1/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE&feature_maps/feature_map_norm2/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_norm2/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv3/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv3/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE&feature_maps/feature_map_norm3/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_norm3/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'feature_maps/feature_map_conv4/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_conv4/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE&feature_maps/feature_map_norm4/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_norm4/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'primary_caps/primary_cap_dconv/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%primary_caps/primary_cap_dconv/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
жг
VARIABLE_VALUE(digit_caps/digit_caps_transform_tensor/vglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЪЧ
VARIABLE_VALUE"digit_caps/digit_caps_log_priors/valayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv1/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv1/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE&feature_maps/feature_map_norm1/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_norm1/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE&feature_maps/feature_map_norm2/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_norm2/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE'feature_maps/feature_map_conv3/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ИЕ
VARIABLE_VALUE%feature_maps/feature_map_conv3/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE&feature_maps/feature_map_norm3/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_norm3/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'feature_maps/feature_map_conv4/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_conv4/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
КЗ
VARIABLE_VALUE&feature_maps/feature_map_norm4/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%feature_maps/feature_map_norm4/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE'primary_caps/primary_cap_dconv/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЙЖ
VARIABLE_VALUE%primary_caps/primary_cap_dconv/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
П
serving_default_input_imagesPlaceholder*/
_output_shapes
:         *
dtype0*$
shape:         
┌
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_images%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variance%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/bias&digit_caps/digit_caps_transform_tensorConst digit_caps/digit_caps_log_priors*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_signature_wrapper_8333
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ъ$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:digit_caps/digit_caps_transform_tensor/Read/ReadVariableOp4digit_caps/digit_caps_log_priors/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp9feature_maps/feature_map_conv1/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv1/bias/Read/ReadVariableOp8feature_maps/feature_map_norm1/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm1/beta/Read/ReadVariableOp9feature_maps/feature_map_conv2/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv2/bias/Read/ReadVariableOp8feature_maps/feature_map_norm2/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm2/beta/Read/ReadVariableOp9feature_maps/feature_map_conv3/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv3/bias/Read/ReadVariableOp8feature_maps/feature_map_norm3/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm3/beta/Read/ReadVariableOp9feature_maps/feature_map_conv4/kernel/Read/ReadVariableOp7feature_maps/feature_map_conv4/bias/Read/ReadVariableOp8feature_maps/feature_map_norm4/gamma/Read/ReadVariableOp7feature_maps/feature_map_norm4/beta/Read/ReadVariableOp9primary_caps/primary_cap_dconv/kernel/Read/ReadVariableOp7primary_caps/primary_cap_dconv/bias/Read/ReadVariableOp>feature_maps/feature_map_norm1/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm1/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm2/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm2/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm3/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm3/moving_variance/Read/ReadVariableOp>feature_maps/feature_map_norm4/moving_mean/Read/ReadVariableOpBfeature_maps/feature_map_norm4/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp<digit_caps/digit_caps_transform_tensor/m/Read/ReadVariableOp6digit_caps/digit_caps_log_priors/m/Read/ReadVariableOp;feature_maps/feature_map_conv1/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv1/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm1/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm1/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv2/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv2/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm2/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm2/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv3/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv3/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm3/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm3/beta/m/Read/ReadVariableOp;feature_maps/feature_map_conv4/kernel/m/Read/ReadVariableOp9feature_maps/feature_map_conv4/bias/m/Read/ReadVariableOp:feature_maps/feature_map_norm4/gamma/m/Read/ReadVariableOp9feature_maps/feature_map_norm4/beta/m/Read/ReadVariableOp;primary_caps/primary_cap_dconv/kernel/m/Read/ReadVariableOp9primary_caps/primary_cap_dconv/bias/m/Read/ReadVariableOp<digit_caps/digit_caps_transform_tensor/v/Read/ReadVariableOp6digit_caps/digit_caps_log_priors/v/Read/ReadVariableOp;feature_maps/feature_map_conv1/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv1/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm1/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm1/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv2/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv2/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm2/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm2/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv3/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv3/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm3/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm3/beta/v/Read/ReadVariableOp;feature_maps/feature_map_conv4/kernel/v/Read/ReadVariableOp9feature_maps/feature_map_conv4/bias/v/Read/ReadVariableOp:feature_maps/feature_map_norm4/gamma/v/Read/ReadVariableOp9feature_maps/feature_map_norm4/beta/v/Read/ReadVariableOp;primary_caps/primary_cap_dconv/kernel/v/Read/ReadVariableOp9primary_caps/primary_cap_dconv/bias/v/Read/ReadVariableOpConst_1*Z
TinS
Q2O	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__traced_save_9942
р
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&digit_caps/digit_caps_transform_tensor digit_caps/digit_caps_log_priors	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate%feature_maps/feature_map_conv1/kernel#feature_maps/feature_map_conv1/bias$feature_maps/feature_map_norm1/gamma#feature_maps/feature_map_norm1/beta%feature_maps/feature_map_conv2/kernel#feature_maps/feature_map_conv2/bias$feature_maps/feature_map_norm2/gamma#feature_maps/feature_map_norm2/beta%feature_maps/feature_map_conv3/kernel#feature_maps/feature_map_conv3/bias$feature_maps/feature_map_norm3/gamma#feature_maps/feature_map_norm3/beta%feature_maps/feature_map_conv4/kernel#feature_maps/feature_map_conv4/bias$feature_maps/feature_map_norm4/gamma#feature_maps/feature_map_norm4/beta%primary_caps/primary_cap_dconv/kernel#primary_caps/primary_cap_dconv/bias*feature_maps/feature_map_norm1/moving_mean.feature_maps/feature_map_norm1/moving_variance*feature_maps/feature_map_norm2/moving_mean.feature_maps/feature_map_norm2/moving_variance*feature_maps/feature_map_norm3/moving_mean.feature_maps/feature_map_norm3/moving_variance*feature_maps/feature_map_norm4/moving_mean.feature_maps/feature_map_norm4/moving_variancetotalcounttotal_1count_1(digit_caps/digit_caps_transform_tensor/m"digit_caps/digit_caps_log_priors/m'feature_maps/feature_map_conv1/kernel/m%feature_maps/feature_map_conv1/bias/m&feature_maps/feature_map_norm1/gamma/m%feature_maps/feature_map_norm1/beta/m'feature_maps/feature_map_conv2/kernel/m%feature_maps/feature_map_conv2/bias/m&feature_maps/feature_map_norm2/gamma/m%feature_maps/feature_map_norm2/beta/m'feature_maps/feature_map_conv3/kernel/m%feature_maps/feature_map_conv3/bias/m&feature_maps/feature_map_norm3/gamma/m%feature_maps/feature_map_norm3/beta/m'feature_maps/feature_map_conv4/kernel/m%feature_maps/feature_map_conv4/bias/m&feature_maps/feature_map_norm4/gamma/m%feature_maps/feature_map_norm4/beta/m'primary_caps/primary_cap_dconv/kernel/m%primary_caps/primary_cap_dconv/bias/m(digit_caps/digit_caps_transform_tensor/v"digit_caps/digit_caps_log_priors/v'feature_maps/feature_map_conv1/kernel/v%feature_maps/feature_map_conv1/bias/v&feature_maps/feature_map_norm1/gamma/v%feature_maps/feature_map_norm1/beta/v'feature_maps/feature_map_conv2/kernel/v%feature_maps/feature_map_conv2/bias/v&feature_maps/feature_map_norm2/gamma/v%feature_maps/feature_map_norm2/beta/v'feature_maps/feature_map_conv3/kernel/v%feature_maps/feature_map_conv3/bias/v&feature_maps/feature_map_norm3/gamma/v%feature_maps/feature_map_norm3/beta/v'feature_maps/feature_map_conv4/kernel/v%feature_maps/feature_map_conv4/bias/v&feature_maps/feature_map_norm4/gamma/v%feature_maps/feature_map_norm4/beta/v'primary_caps/primary_cap_dconv/kernel/v%primary_caps/primary_cap_dconv/bias/v*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_restore_10183юб
▌
Ц
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_6890

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
С
║
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9563

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╒.
╧
F__inference_primary_caps_layer_call_and_return_conditional_losses_7424
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		А@
1primary_cap_dconv_biasadd_readvariableop_resource:	А
identityИв(primary_cap_dconv/BiasAdd/ReadVariableOpв'primary_cap_dconv/Conv2D/ReadVariableOp╠
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		А*
dtype02)
'primary_cap_dconv/Conv2D/ReadVariableOpс
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2
primary_cap_dconv/Conv2D├
(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(primary_cap_dconv/BiasAdd/ReadVariableOp╤
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
primary_cap_dconv/BiasAddЧ
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
primary_cap_dconv/ReluК
primary_cap_reshape/ShapeShape$primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:2
primary_cap_reshape/ShapeЬ
'primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'primary_cap_reshape/strided_slice/stackа
)primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)primary_cap_reshape/strided_slice/stack_1а
)primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)primary_cap_reshape/strided_slice/stack_2┌
!primary_cap_reshape/strided_sliceStridedSlice"primary_cap_reshape/Shape:output:00primary_cap_reshape/strided_slice/stack:output:02primary_cap_reshape/strided_slice/stack_1:output:02primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!primary_cap_reshape/strided_sliceХ
#primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2%
#primary_cap_reshape/Reshape/shape/1М
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#primary_cap_reshape/Reshape/shape/2Д
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!primary_cap_reshape/Reshape/shape═
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
primary_cap_reshape/Reshape├
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/norm/mul▒
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2/
-primary_cap_squash/norm/Sum/reduction_indicesс
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         *
	keep_dims(2
primary_cap_squash/norm/Sumа
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/norm/SqrtП
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         2
primary_cap_squash/ExpБ
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
primary_cap_squash/truediv/x╝
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         2
primary_cap_squash/truedivy
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
primary_cap_squash/sub/x░
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/suby
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32
primary_cap_squash/add/y┤
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/add┐
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/truediv_1л
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/muly
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:         2

Identityг
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         		А
&
_user_specified_namefeature_maps
┐	
╧
0__inference_feature_map_norm4_layer_call_fn_9638

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_71422
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╖	
╦
0__inference_feature_map_norm3_layer_call_fn_9576

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_70162
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
▌
Ц
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9607

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
С
║
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9501

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
■y
щ
F__inference_feature_maps_layer_call_and_return_conditional_losses_9143
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@А@
1feature_map_conv4_biasadd_readvariableop_resource:	А8
)feature_map_norm4_readvariableop_resource:	А:
+feature_map_norm4_readvariableop_1_resource:	АI
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АK
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв(feature_map_conv1/BiasAdd/ReadVariableOpв'feature_map_conv1/Conv2D/ReadVariableOpв(feature_map_conv2/BiasAdd/ReadVariableOpв'feature_map_conv2/Conv2D/ReadVariableOpв(feature_map_conv3/BiasAdd/ReadVariableOpв'feature_map_conv3/Conv2D/ReadVariableOpв(feature_map_conv4/BiasAdd/ReadVariableOpв'feature_map_conv4/Conv2D/ReadVariableOpв1feature_map_norm1/FusedBatchNormV3/ReadVariableOpв3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm1/ReadVariableOpв"feature_map_norm1/ReadVariableOp_1в1feature_map_norm2/FusedBatchNormV3/ReadVariableOpв3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm2/ReadVariableOpв"feature_map_norm2/ReadVariableOp_1в1feature_map_norm3/FusedBatchNormV3/ReadVariableOpв3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm3/ReadVariableOpв"feature_map_norm3/ReadVariableOp_1в1feature_map_norm4/FusedBatchNormV3/ReadVariableOpв3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm4/ReadVariableOpв"feature_map_norm4/ReadVariableOp_1╦
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'feature_map_conv1/Conv2D/ReadVariableOpр
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
feature_map_conv1/Conv2D┬
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(feature_map_conv1/BiasAdd/ReadVariableOp╨
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
feature_map_conv1/BiasAddЦ
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
feature_map_conv1/Reluк
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02"
 feature_map_norm1/ReadVariableOp░
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"feature_map_norm1/ReadVariableOp_1▌
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2$
"feature_map_norm1/FusedBatchNormV3╦
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'feature_map_conv2/Conv2D/ReadVariableOp·
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv2/Conv2D┬
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv2/BiasAdd/ReadVariableOp╨
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/BiasAddЦ
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/Reluк
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm2/ReadVariableOp░
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm2/ReadVariableOp_1▌
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm2/FusedBatchNormV3╦
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'feature_map_conv3/Conv2D/ReadVariableOp·
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv3/Conv2D┬
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv3/BiasAdd/ReadVariableOp╨
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/BiasAddЦ
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/Reluк
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm3/ReadVariableOp░
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm3/ReadVariableOp_1▌
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm3/FusedBatchNormV3╠
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02)
'feature_map_conv4/Conv2D/ReadVariableOp√
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2
feature_map_conv4/Conv2D├
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(feature_map_conv4/BiasAdd/ReadVariableOp╤
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/BiasAddЧ
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/Reluл
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 feature_map_norm4/ReadVariableOp▒
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02$
"feature_map_norm4/ReadVariableOp_1▐
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype023
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpф
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype025
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1┘
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm4/FusedBatchNormV3К
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         		А2

Identityъ
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
Х
a
E__inference_digit_probs_layer_call_and_return_conditional_losses_9430

inputs
identitya
norm/mulMulinputsinputs*
T0*+
_output_shapes
:         
2

norm/mulЛ
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2
norm/Sum/reduction_indicesХ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2

norm/Sumg
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:         
2
	norm/SqrtИ
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
norm/Squeezei
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
б
╛
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9687

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
Г'
г
digit_caps_map_while_body_8594:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_19
5digit_caps_map_while_digit_caps_map_strided_slice_1_0u
qdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0O
5digit_caps_map_while_matmul_readvariableop_resource_0:
!
digit_caps_map_while_identity#
digit_caps_map_while_identity_1#
digit_caps_map_while_identity_2#
digit_caps_map_while_identity_37
3digit_caps_map_while_digit_caps_map_strided_slice_1s
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorM
3digit_caps_map_while_matmul_readvariableop_resource:
Ив*digit_caps/map/while/MatMul/ReadVariableOpщ
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2H
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeм
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0 digit_caps_map_while_placeholderOdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype02:
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItem╓
*digit_caps/map/while/MatMul/ReadVariableOpReadVariableOp5digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype02,
*digit_caps/map/while/MatMul/ReadVariableOpё
digit_caps/map/while/MatMulBatchMatMulV22digit_caps/map/while/MatMul/ReadVariableOp:value:0?digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
2
digit_caps/map/while/MatMulд
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"digit_caps_map_while_placeholder_1 digit_caps_map_while_placeholder$digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype02;
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemz
digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/map/while/add/yе
digit_caps/map/while/addAddV2 digit_caps_map_while_placeholder#digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: 2
digit_caps/map/while/add~
digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/map/while/add_1/y┴
digit_caps/map/while/add_1AddV26digit_caps_map_while_digit_caps_map_while_loop_counter%digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/map/while/add_1з
digit_caps/map/while/IdentityIdentitydigit_caps/map/while/add_1:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2
digit_caps/map/while/Identity╛
digit_caps/map/while/Identity_1Identity1digit_caps_map_while_digit_caps_map_strided_slice^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_1й
digit_caps/map/while/Identity_2Identitydigit_caps/map/while/add:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_2╓
digit_caps/map/while/Identity_3IdentityIdigit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_3е
digit_caps/map/while/NoOpNoOp+^digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
digit_caps/map/while/NoOp"l
3digit_caps_map_while_digit_caps_map_strided_slice_15digit_caps_map_while_digit_caps_map_strided_slice_1_0"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0"K
digit_caps_map_while_identity_1(digit_caps/map/while/Identity_1:output:0"K
digit_caps_map_while_identity_2(digit_caps/map/while/Identity_2:output:0"K
digit_caps_map_while_identity_3(digit_caps/map/while/Identity_3:output:0"l
3digit_caps_map_while_matmul_readvariableop_resource5digit_caps_map_while_matmul_readvariableop_resource_0"ф
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2X
*digit_caps/map/while/MatMul/ReadVariableOp*digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
с
╗
)__inference_digit_caps_layer_call_fn_9284
primary_caps!
unknown:

	unknown_0
	unknown_1:

identityИвStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallprimary_capsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_digit_caps_layer_call_and_return_conditional_losses_75572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:         
&
_user_specified_nameprimary_caps:

_output_shapes
: 
Х
a
E__inference_digit_probs_layer_call_and_return_conditional_losses_7655

inputs
identitya
norm/mulMulinputsinputs*
T0*+
_output_shapes
:         
2

norm/mulЛ
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2
norm/Sum/reduction_indicesХ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2

norm/Sumg
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:         
2
	norm/SqrtИ
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
norm/Squeezei
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
В
ф
digit_caps_map_while_cond_8838:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_1:
6digit_caps_map_while_less_digit_caps_map_strided_sliceP
Ldigit_caps_map_while_digit_caps_map_while_cond_8838___redundant_placeholder0P
Ldigit_caps_map_while_digit_caps_map_while_cond_8838___redundant_placeholder1!
digit_caps_map_while_identity
╣
digit_caps/map/while/LessLess digit_caps_map_while_placeholder6digit_caps_map_while_less_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2
digit_caps/map/while/Less╬
digit_caps/map/while/Less_1Less6digit_caps_map_while_digit_caps_map_while_loop_counter1digit_caps_map_while_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2
digit_caps/map/while/Less_1и
digit_caps/map/while/LogicalAnd
LogicalAnddigit_caps/map/while/Less_1:z:0digit_caps/map/while/Less:z:0*
_output_shapes
: 2!
digit_caps/map/while/LogicalAndР
digit_caps/map/while/IdentityIdentity#digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
digit_caps/map/while/Identity"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Ь[
∙
D__inference_digit_caps_layer_call_and_return_conditional_losses_9411
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identityИвadd_3/ReadVariableOpв	map/whileb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimЗ

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:         2

ExpandDimsy
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         2
Tile/multiples|
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:         
2
Tilew
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
digit_cap_inputs/dimЮ
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:         
2
digit_cap_inputs_
	map/ShapeShapedigit_cap_inputs:output:0*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stackА
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1А
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2·
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_sliceН
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2!
map/TensorArrayV2/element_shape└
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2╧
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeР
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/ConstС
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2#
!map/TensorArrayV2_1/element_shape╞
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counterХ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
map_while_body_9306*
condR
map_while_cond_9305*!
output_shapes
: : : : : : : 2
	map/while┼
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            26
4map/TensorArrayV2Stack/TensorListStack/element_shapeА
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:         
*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStack─
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:         
*
squeeze_dims

         2
digit_cap_predictions─
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
*
adj_y(2
digit_cap_attentionsq
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:         
2
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        2
Sum/reduction_indicesЕ
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:         
*
	keep_dims(2
SumN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankY
add/xConst*
_output_shapes
: *
dtype0*
valueB :
■        2
add/xS
addAddV2add/x:output:0Rank:output:0*
T0*
_output_shapes
: 2
addR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1P
mod/yConst*
_output_shapes
: *
dtype0*
value	B :2
mod/yP
modFloorModadd:z:0mod/y:output:0*
T0*
_output_shapes
: 2
modP
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
Sub/yS
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: 2
Sub\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltah
rangeRangerange/start:output:0mod:z:0range/delta:output:0*
_output_shapes
:2
rangeT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yS
add_1AddV2mod:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltaa
range_1Range	add_1:z:0Sub:z:0range_1/delta:output:0*
_output_shapes
: 2	
range_1a
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:2
concat/values_1a
concat/values_3Packmod:z:0*
N*
T0*
_output_shapes
:2
concat/values_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╢
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat|
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:         
2
	transposef
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:         
2	
SoftmaxT
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Sub_1/yY
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: 2
Sub_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/start`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/deltap
range_2Rangerange_2/start:output:0mod:z:0range_2/delta:output:0*
_output_shapes
:2	
range_2T
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yS
add_2AddV2mod:z:0add_2/y:output:0*
T0*
_output_shapes
: 2
add_2`
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/deltac
range_3Range	add_2:z:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: 2	
range_3g
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_1e
concat_1/values_3Packmod:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_3`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis┬
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1п
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:         
2!
digit_cap_coupling_coefficientsО
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype02
add_3/ReadVariableOpФ
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
add_3Ж
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
2
MatMulД
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

■        2	
SqueezeЧ
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/norm/mulн
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+digit_cap_squash/norm/Sum/reduction_indices┘
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2
digit_cap_squash/norm/SumЪ
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/norm/SqrtЙ
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/Exp}
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
digit_cap_squash/truediv/x┤
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/truedivu
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
digit_cap_squash/sub/xи
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/subu
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32
digit_cap_squash/add/yм
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/addе
digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/truediv_1г
digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/mulw
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:         
2

Identityq
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:         
&
_user_specified_nameprimary_caps:

_output_shapes
: 
с
Г
+__inference_feature_maps_layer_call_fn_9055
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_78242
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
Фз
║)
__inference__traced_save_9942
file_prefixE
Asavev2_digit_caps_digit_caps_transform_tensor_read_readvariableop?
;savev2_digit_caps_digit_caps_log_priors_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv1_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm1_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm1_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv2_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm2_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm2_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv3_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm3_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm3_beta_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_kernel_read_readvariableopB
>savev2_feature_maps_feature_map_conv4_bias_read_readvariableopC
?savev2_feature_maps_feature_map_norm4_gamma_read_readvariableopB
>savev2_feature_maps_feature_map_norm4_beta_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_kernel_read_readvariableopB
>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableopI
Esavev2_feature_maps_feature_map_norm1_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm1_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm2_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm2_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm3_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm3_moving_variance_read_readvariableopI
Esavev2_feature_maps_feature_map_norm4_moving_mean_read_readvariableopM
Isavev2_feature_maps_feature_map_norm4_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopG
Csavev2_digit_caps_digit_caps_transform_tensor_m_read_readvariableopA
=savev2_digit_caps_digit_caps_log_priors_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv1_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm1_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm1_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv2_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm2_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm2_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv3_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm3_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm3_beta_m_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv4_kernel_m_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_bias_m_read_readvariableopE
Asavev2_feature_maps_feature_map_norm4_gamma_m_read_readvariableopD
@savev2_feature_maps_feature_map_norm4_beta_m_read_readvariableopF
Bsavev2_primary_caps_primary_cap_dconv_kernel_m_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_bias_m_read_readvariableopG
Csavev2_digit_caps_digit_caps_transform_tensor_v_read_readvariableopA
=savev2_digit_caps_digit_caps_log_priors_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv1_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv1_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm1_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm1_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv2_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv2_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm2_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm2_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv3_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv3_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm3_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm3_beta_v_read_readvariableopF
Bsavev2_feature_maps_feature_map_conv4_kernel_v_read_readvariableopD
@savev2_feature_maps_feature_map_conv4_bias_v_read_readvariableopE
Asavev2_feature_maps_feature_map_norm4_gamma_v_read_readvariableopD
@savev2_feature_maps_feature_map_norm4_beta_v_read_readvariableopF
Bsavev2_primary_caps_primary_cap_dconv_kernel_v_read_readvariableopD
@savev2_primary_caps_primary_cap_dconv_bias_v_read_readvariableop
savev2_const_1

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameц(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*°'
valueю'Bы'NBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesз
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueзBдNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesг(
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_digit_caps_digit_caps_transform_tensor_read_readvariableop;savev2_digit_caps_digit_caps_log_priors_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop@savev2_feature_maps_feature_map_conv1_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv1_bias_read_readvariableop?savev2_feature_maps_feature_map_norm1_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm1_beta_read_readvariableop@savev2_feature_maps_feature_map_conv2_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv2_bias_read_readvariableop?savev2_feature_maps_feature_map_norm2_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm2_beta_read_readvariableop@savev2_feature_maps_feature_map_conv3_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv3_bias_read_readvariableop?savev2_feature_maps_feature_map_norm3_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm3_beta_read_readvariableop@savev2_feature_maps_feature_map_conv4_kernel_read_readvariableop>savev2_feature_maps_feature_map_conv4_bias_read_readvariableop?savev2_feature_maps_feature_map_norm4_gamma_read_readvariableop>savev2_feature_maps_feature_map_norm4_beta_read_readvariableop@savev2_primary_caps_primary_cap_dconv_kernel_read_readvariableop>savev2_primary_caps_primary_cap_dconv_bias_read_readvariableopEsavev2_feature_maps_feature_map_norm1_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm1_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm2_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm2_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm3_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm3_moving_variance_read_readvariableopEsavev2_feature_maps_feature_map_norm4_moving_mean_read_readvariableopIsavev2_feature_maps_feature_map_norm4_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopCsavev2_digit_caps_digit_caps_transform_tensor_m_read_readvariableop=savev2_digit_caps_digit_caps_log_priors_m_read_readvariableopBsavev2_feature_maps_feature_map_conv1_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv1_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm1_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm1_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv2_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv2_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm2_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm2_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv3_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv3_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm3_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm3_beta_m_read_readvariableopBsavev2_feature_maps_feature_map_conv4_kernel_m_read_readvariableop@savev2_feature_maps_feature_map_conv4_bias_m_read_readvariableopAsavev2_feature_maps_feature_map_norm4_gamma_m_read_readvariableop@savev2_feature_maps_feature_map_norm4_beta_m_read_readvariableopBsavev2_primary_caps_primary_cap_dconv_kernel_m_read_readvariableop@savev2_primary_caps_primary_cap_dconv_bias_m_read_readvariableopCsavev2_digit_caps_digit_caps_transform_tensor_v_read_readvariableop=savev2_digit_caps_digit_caps_log_priors_v_read_readvariableopBsavev2_feature_maps_feature_map_conv1_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv1_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm1_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm1_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv2_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv2_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm2_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm2_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv3_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv3_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm3_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm3_beta_v_read_readvariableopBsavev2_feature_maps_feature_map_conv4_kernel_v_read_readvariableop@savev2_feature_maps_feature_map_conv4_bias_v_read_readvariableopAsavev2_feature_maps_feature_map_norm4_gamma_v_read_readvariableop@savev2_feature_maps_feature_map_norm4_beta_v_read_readvariableopBsavev2_primary_caps_primary_cap_dconv_kernel_v_read_readvariableop@savev2_primary_caps_primary_cap_dconv_bias_v_read_readvariableopsavev2_const_1"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*╟
_input_shapes╡
▓: :
:
: : : : : : : : : : @:@:@:@:@@:@:@:@:@А:А:А:А:		А:А: : :@:@:@:@:А:А: : : : :
:
: : : : : @:@:@:@:@@:@:@:@:@А:А:А:А:		А:А:
:
: : : : : @:@:@:@:@@:@:@:@:@А:А:А:А:		А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:
:($
"
_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:-)
'
_output_shapes
:		А:!

_output_shapes	
:А: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:! 

_output_shapes	
:А:!!

_output_shapes	
:А:"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
:
:('$
"
_output_shapes
:
:,((
&
_output_shapes
: : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
: @: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:-4)
'
_output_shapes
:@А:!5

_output_shapes	
:А:!6

_output_shapes	
:А:!7

_output_shapes	
:А:-8)
'
_output_shapes
:		А:!9

_output_shapes	
:А:,:(
&
_output_shapes
:
:(;$
"
_output_shapes
:
:,<(
&
_output_shapes
: : =

_output_shapes
: : >

_output_shapes
: : ?

_output_shapes
: :,@(
&
_output_shapes
: @: A

_output_shapes
:@: B

_output_shapes
:@: C

_output_shapes
:@:,D(
&
_output_shapes
:@@: E

_output_shapes
:@: F

_output_shapes
:@: G

_output_shapes
:@:-H)
'
_output_shapes
:@А:!I

_output_shapes	
:А:!J

_output_shapes	
:А:!K

_output_shapes	
:А:-L)
'
_output_shapes
:		А:!M

_output_shapes	
:А:N

_output_shapes
: 
С
║
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9625

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╒
Ь
0__inference_Efficient-CapsNet_layer_call_fn_8396

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А%

unknown_23:		А

unknown_24:	А$

unknown_25:


unknown_26 

unknown_27:

identityИвStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_75772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
╒.
╧
F__inference_primary_caps_layer_call_and_return_conditional_losses_9273
feature_mapsK
0primary_cap_dconv_conv2d_readvariableop_resource:		А@
1primary_cap_dconv_biasadd_readvariableop_resource:	А
identityИв(primary_cap_dconv/BiasAdd/ReadVariableOpв'primary_cap_dconv/Conv2D/ReadVariableOp╠
'primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp0primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		А*
dtype02)
'primary_cap_dconv/Conv2D/ReadVariableOpс
primary_cap_dconv/Conv2DConv2Dfeature_maps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2
primary_cap_dconv/Conv2D├
(primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp1primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(primary_cap_dconv/BiasAdd/ReadVariableOp╤
primary_cap_dconv/BiasAddBiasAdd!primary_cap_dconv/Conv2D:output:00primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2
primary_cap_dconv/BiasAddЧ
primary_cap_dconv/ReluRelu"primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:         А2
primary_cap_dconv/ReluК
primary_cap_reshape/ShapeShape$primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:2
primary_cap_reshape/ShapeЬ
'primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'primary_cap_reshape/strided_slice/stackа
)primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)primary_cap_reshape/strided_slice/stack_1а
)primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)primary_cap_reshape/strided_slice/stack_2┌
!primary_cap_reshape/strided_sliceStridedSlice"primary_cap_reshape/Shape:output:00primary_cap_reshape/strided_slice/stack:output:02primary_cap_reshape/strided_slice/stack_1:output:02primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!primary_cap_reshape/strided_sliceХ
#primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2%
#primary_cap_reshape/Reshape/shape/1М
#primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2%
#primary_cap_reshape/Reshape/shape/2Д
!primary_cap_reshape/Reshape/shapePack*primary_cap_reshape/strided_slice:output:0,primary_cap_reshape/Reshape/shape/1:output:0,primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2#
!primary_cap_reshape/Reshape/shape═
primary_cap_reshape/ReshapeReshape$primary_cap_dconv/Relu:activations:0*primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2
primary_cap_reshape/Reshape├
primary_cap_squash/norm/mulMul$primary_cap_reshape/Reshape:output:0$primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/norm/mul▒
-primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2/
-primary_cap_squash/norm/Sum/reduction_indicesс
primary_cap_squash/norm/SumSumprimary_cap_squash/norm/mul:z:06primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         *
	keep_dims(2
primary_cap_squash/norm/Sumа
primary_cap_squash/norm/SqrtSqrt$primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/norm/SqrtП
primary_cap_squash/ExpExp primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         2
primary_cap_squash/ExpБ
primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
primary_cap_squash/truediv/x╝
primary_cap_squash/truedivRealDiv%primary_cap_squash/truediv/x:output:0primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         2
primary_cap_squash/truedivy
primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
primary_cap_squash/sub/x░
primary_cap_squash/subSub!primary_cap_squash/sub/x:output:0primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/suby
primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32
primary_cap_squash/add/y┤
primary_cap_squash/addAddV2 primary_cap_squash/norm/Sqrt:y:0!primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         2
primary_cap_squash/add┐
primary_cap_squash/truediv_1RealDiv$primary_cap_reshape/Reshape:output:0primary_cap_squash/add:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/truediv_1л
primary_cap_squash/mulMulprimary_cap_squash/sub:z:0 primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         2
primary_cap_squash/muly
IdentityIdentityprimary_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:         2

Identityг
NoOpNoOp)^primary_cap_dconv/BiasAdd/ReadVariableOp(^primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 2T
(primary_cap_dconv/BiasAdd/ReadVariableOp(primary_cap_dconv/BiasAdd/ReadVariableOp2R
'primary_cap_dconv/Conv2D/ReadVariableOp'primary_cap_dconv/Conv2D/ReadVariableOp:^ Z
0
_output_shapes
:         		А
&
_user_specified_namefeature_maps
▌
Ц
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_7016

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
С
║
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_6808

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╖	
╦
0__inference_feature_map_norm2_layer_call_fn_9514

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_68902
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
С
║
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_6934

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
С
║
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_7060

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ъ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Х
a
E__inference_digit_probs_layer_call_and_return_conditional_losses_7574

inputs
identitya
norm/mulMulinputsinputs*
T0*+
_output_shapes
:         
2

norm/mulЛ
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2
norm/Sum/reduction_indicesХ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2

norm/Sumg
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:         
2
	norm/SqrtИ
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
norm/Squeezei
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
╡	
╦
0__inference_feature_map_norm2_layer_call_fn_9527

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_69342
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ч
в
0__inference_Efficient-CapsNet_layer_call_fn_7638
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А%

unknown_23:		А

unknown_24:	А$

unknown_25:


unknown_26 

unknown_27:

identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_75772
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
╝4
│

0Efficient-CapsNet_digit_caps_map_while_body_6632^
Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterY
Uefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice6
2efficient_capsnet_digit_caps_map_while_placeholder8
4efficient_capsnet_digit_caps_map_while_placeholder_1]
Yefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1_0Ъ
Хefficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0a
Gefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0:
3
/efficient_capsnet_digit_caps_map_while_identity5
1efficient_capsnet_digit_caps_map_while_identity_15
1efficient_capsnet_digit_caps_map_while_identity_25
1efficient_capsnet_digit_caps_map_while_identity_3[
Wefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1Ш
Уefficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_
Eefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource:
Ив<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOpН
XEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2Z
XEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeЩ
JEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemХefficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_02efficient_capsnet_digit_caps_map_while_placeholderaEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype02L
JEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItemМ
<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOpReadVariableOpGefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype02>
<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp╣
-Efficient-CapsNet/digit_caps/map/while/MatMulBatchMatMulV2DEfficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp:value:0QEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
2/
-Efficient-CapsNet/digit_caps/map/while/MatMul■
KEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem4efficient_capsnet_digit_caps_map_while_placeholder_12efficient_capsnet_digit_caps_map_while_placeholder6Efficient-CapsNet/digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype02M
KEfficient-CapsNet/digit_caps/map/while/TensorArrayV2Write/TensorListSetItemЮ
,Efficient-CapsNet/digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,Efficient-CapsNet/digit_caps/map/while/add/yэ
*Efficient-CapsNet/digit_caps/map/while/addAddV22efficient_capsnet_digit_caps_map_while_placeholder5Efficient-CapsNet/digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: 2,
*Efficient-CapsNet/digit_caps/map/while/addв
.Efficient-CapsNet/digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :20
.Efficient-CapsNet/digit_caps/map/while/add_1/yЫ
,Efficient-CapsNet/digit_caps/map/while/add_1AddV2Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counter7Efficient-CapsNet/digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2.
,Efficient-CapsNet/digit_caps/map/while/add_1я
/Efficient-CapsNet/digit_caps/map/while/IdentityIdentity0Efficient-CapsNet/digit_caps/map/while/add_1:z:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: 21
/Efficient-CapsNet/digit_caps/map/while/IdentityШ
1Efficient-CapsNet/digit_caps/map/while/Identity_1IdentityUefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: 23
1Efficient-CapsNet/digit_caps/map/while/Identity_1ё
1Efficient-CapsNet/digit_caps/map/while/Identity_2Identity.Efficient-CapsNet/digit_caps/map/while/add:z:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: 23
1Efficient-CapsNet/digit_caps/map/while/Identity_2Ю
1Efficient-CapsNet/digit_caps/map/while/Identity_3Identity[Efficient-CapsNet/digit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0,^Efficient-CapsNet/digit_caps/map/while/NoOp*
T0*
_output_shapes
: 23
1Efficient-CapsNet/digit_caps/map/while/Identity_3█
+Efficient-CapsNet/digit_caps/map/while/NoOpNoOp=^Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2-
+Efficient-CapsNet/digit_caps/map/while/NoOp"┤
Wefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1Yefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice_1_0"k
/efficient_capsnet_digit_caps_map_while_identity8Efficient-CapsNet/digit_caps/map/while/Identity:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_1:Efficient-CapsNet/digit_caps/map/while/Identity_1:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_2:Efficient-CapsNet/digit_caps/map/while/Identity_2:output:0"o
1efficient_capsnet_digit_caps_map_while_identity_3:Efficient-CapsNet/digit_caps/map/while/Identity_3:output:0"Р
Eefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resourceGefficient_capsnet_digit_caps_map_while_matmul_readvariableop_resource_0"о
Уefficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensorХefficient_capsnet_digit_caps_map_while_tensorarrayv2read_tensorlistgetitem_efficient_capsnet_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2|
<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp<Efficient-CapsNet/digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
о
и
+__inference_primary_caps_layer_call_fn_9240
feature_maps"
unknown:		А
	unknown_0:	А
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallfeature_mapsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_primary_caps_layer_call_and_return_conditional_losses_74242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :         		А: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
0
_output_shapes
:         		А
&
_user_specified_namefeature_maps
═
Ь
0__inference_Efficient-CapsNet_layer_call_fn_8459

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А%

unknown_23:		А

unknown_24:	А$

unknown_25:


unknown_26 

unknown_27:

identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_80062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
╕Щ
У&
__inference__wrapped_model_6742
input_imagesi
Oefficient_capsnet_feature_maps_feature_map_conv1_conv2d_readvariableop_resource: ^
Pefficient_capsnet_feature_maps_feature_map_conv1_biasadd_readvariableop_resource: V
Hefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_resource: X
Jefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_1_resource: g
Yefficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: i
[efficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: i
Oefficient_capsnet_feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @^
Pefficient_capsnet_feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@V
Hefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_resource:@X
Jefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_1_resource:@g
Yefficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@i
[efficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@i
Oefficient_capsnet_feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@^
Pefficient_capsnet_feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@V
Hefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_resource:@X
Jefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_1_resource:@g
Yefficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@i
[efficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@j
Oefficient_capsnet_feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@А_
Pefficient_capsnet_feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	АW
Hefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_resource:	АY
Jefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_1_resource:	Аh
Yefficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	Аj
[efficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	Аj
Oefficient_capsnet_primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		А_
Pefficient_capsnet_primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	АH
.efficient_capsnet_digit_caps_map_while_input_6:
&
"efficient_capsnet_digit_caps_mul_xP
:efficient_capsnet_digit_caps_add_3_readvariableop_resource:

identityИв1Efficient-CapsNet/digit_caps/add_3/ReadVariableOpв&Efficient-CapsNet/digit_caps/map/whileвGEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpвFEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpвGEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpвFEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpвGEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpвFEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpвGEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpвFEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpвPEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpвREfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpвAEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1вPEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpвREfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpвAEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1вPEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpвREfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpвAEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1вPEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpвREfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpвAEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1вGEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpвFEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpи
FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02H
FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp╜
7Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2DConv2Dinput_imagesNEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
29
7Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2DЯ
GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02I
GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp╠
8Efficient-CapsNet/feature_maps/feature_map_conv1/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2:
8Efficient-CapsNet/feature_maps/feature_map_conv1/BiasAddє
5Efficient-CapsNet/feature_maps/feature_map_conv1/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          27
5Efficient-CapsNet/feature_maps/feature_map_conv1/ReluЗ
?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02A
?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpН
AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype02C
AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1║
PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02R
PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp└
REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02T
REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1н
AEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv1/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2C
AEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3и
FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02H
FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpЎ
7Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
29
7Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2DЯ
GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02I
GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp╠
8Efficient-CapsNet/feature_maps/feature_map_conv2/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2:
8Efficient-CapsNet/feature_maps/feature_map_conv2/BiasAddє
5Efficient-CapsNet/feature_maps/feature_map_conv2/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @27
5Efficient-CapsNet/feature_maps/feature_map_conv2/ReluЗ
?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02A
?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpН
AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype02C
AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1║
PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02R
PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp└
REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02T
REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1н
AEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv2/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2C
AEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3и
FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02H
FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpЎ
7Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
29
7Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2DЯ
GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02I
GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp╠
8Efficient-CapsNet/feature_maps/feature_map_conv3/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2:
8Efficient-CapsNet/feature_maps/feature_map_conv3/BiasAddє
5Efficient-CapsNet/feature_maps/feature_map_conv3/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @27
5Efficient-CapsNet/feature_maps/feature_map_conv3/ReluЗ
?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02A
?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpН
AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype02C
AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1║
PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02R
PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp└
REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02T
REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1н
AEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv3/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2C
AEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3й
FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02H
FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpў
7Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3:y:0NEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
29
7Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2Dа
GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02I
GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp═
8Efficient-CapsNet/feature_maps/feature_map_conv4/BiasAddBiasAdd@Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2D:output:0OEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2:
8Efficient-CapsNet/feature_maps/feature_map_conv4/BiasAddЇ
5Efficient-CapsNet/feature_maps/feature_map_conv4/ReluReluAEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А27
5Efficient-CapsNet/feature_maps/feature_map_conv4/ReluИ
?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpReadVariableOpHefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02A
?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpО
AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOpJefficient_capsnet_feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02C
AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1╗
PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpYefficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02R
PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp┴
REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp[efficient_capsnet_feature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02T
REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1▓
AEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV3CEfficient-CapsNet/feature_maps/feature_map_conv4/Relu:activations:0GEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp:value:0IEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1:value:0XEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0ZEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
is_training( 2C
AEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3й
FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOpOefficient_capsnet_primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		А*
dtype02H
FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpў
7Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2DConv2DEEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3:y:0NEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
29
7Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2Dа
GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOpPefficient_capsnet_primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02I
GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp═
8Efficient-CapsNet/primary_caps/primary_cap_dconv/BiasAddBiasAdd@Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D:output:0OEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2:
8Efficient-CapsNet/primary_caps/primary_cap_dconv/BiasAddЇ
5Efficient-CapsNet/primary_caps/primary_cap_dconv/ReluReluAEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:         А27
5Efficient-CapsNet/primary_caps/primary_cap_dconv/Reluч
8Efficient-CapsNet/primary_caps/primary_cap_reshape/ShapeShapeCEfficient-CapsNet/primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:2:
8Efficient-CapsNet/primary_caps/primary_cap_reshape/Shape┌
FEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
FEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack▐
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_1▐
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
HEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_2Ф
@Efficient-CapsNet/primary_caps/primary_cap_reshape/strided_sliceStridedSliceAEfficient-CapsNet/primary_caps/primary_cap_reshape/Shape:output:0OEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack:output:0QEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0QEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2B
@Efficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice╙
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         2D
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/1╩
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2D
BEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/2Я
@Efficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shapePackIEfficient-CapsNet/primary_caps/primary_cap_reshape/strided_slice:output:0KEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/1:output:0KEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2B
@Efficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape╔
:Efficient-CapsNet/primary_caps/primary_cap_reshape/ReshapeReshapeCEfficient-CapsNet/primary_caps/primary_cap_dconv/Relu:activations:0IEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2<
:Efficient-CapsNet/primary_caps/primary_cap_reshape/Reshape┐
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/mulMulCEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:0CEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:         2<
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/mulя
LEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2N
LEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum/reduction_indices▌
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/SumSum>Efficient-CapsNet/primary_caps/primary_cap_squash/norm/mul:z:0UEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         *
	keep_dims(2<
:Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum¤
;Efficient-CapsNet/primary_caps/primary_cap_squash/norm/SqrtSqrtCEfficient-CapsNet/primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         2=
;Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sqrtь
5Efficient-CapsNet/primary_caps/primary_cap_squash/ExpExp?Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         27
5Efficient-CapsNet/primary_caps/primary_cap_squash/Exp┐
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2=
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv/x╕
9Efficient-CapsNet/primary_caps/primary_cap_squash/truedivRealDivDEfficient-CapsNet/primary_caps/primary_cap_squash/truediv/x:output:09Efficient-CapsNet/primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         2;
9Efficient-CapsNet/primary_caps/primary_cap_squash/truediv╖
7Efficient-CapsNet/primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7Efficient-CapsNet/primary_caps/primary_cap_squash/sub/xм
5Efficient-CapsNet/primary_caps/primary_cap_squash/subSub@Efficient-CapsNet/primary_caps/primary_cap_squash/sub/x:output:0=Efficient-CapsNet/primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         27
5Efficient-CapsNet/primary_caps/primary_cap_squash/sub╖
7Efficient-CapsNet/primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓329
7Efficient-CapsNet/primary_caps/primary_cap_squash/add/y░
5Efficient-CapsNet/primary_caps/primary_cap_squash/addAddV2?Efficient-CapsNet/primary_caps/primary_cap_squash/norm/Sqrt:y:0@Efficient-CapsNet/primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         27
5Efficient-CapsNet/primary_caps/primary_cap_squash/add╗
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv_1RealDivCEfficient-CapsNet/primary_caps/primary_cap_reshape/Reshape:output:09Efficient-CapsNet/primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:         2=
;Efficient-CapsNet/primary_caps/primary_cap_squash/truediv_1з
5Efficient-CapsNet/primary_caps/primary_cap_squash/mulMul9Efficient-CapsNet/primary_caps/primary_cap_squash/sub:z:0?Efficient-CapsNet/primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         27
5Efficient-CapsNet/primary_caps/primary_cap_squash/mulЬ
+Efficient-CapsNet/digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2-
+Efficient-CapsNet/digit_caps/ExpandDims/dimЛ
'Efficient-CapsNet/digit_caps/ExpandDims
ExpandDims9Efficient-CapsNet/primary_caps/primary_cap_squash/mul:z:04Efficient-CapsNet/digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2)
'Efficient-CapsNet/digit_caps/ExpandDims│
+Efficient-CapsNet/digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         2-
+Efficient-CapsNet/digit_caps/Tile/multiplesЁ
!Efficient-CapsNet/digit_caps/TileTile0Efficient-CapsNet/digit_caps/ExpandDims:output:04Efficient-CapsNet/digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:         
2#
!Efficient-CapsNet/digit_caps/Tile▒
1Efficient-CapsNet/digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
         23
1Efficient-CapsNet/digit_caps/digit_cap_inputs/dimТ
-Efficient-CapsNet/digit_caps/digit_cap_inputs
ExpandDims*Efficient-CapsNet/digit_caps/Tile:output:0:Efficient-CapsNet/digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:         
2/
-Efficient-CapsNet/digit_caps/digit_cap_inputs╢
&Efficient-CapsNet/digit_caps/map/ShapeShape6Efficient-CapsNet/digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:2(
&Efficient-CapsNet/digit_caps/map/Shape╢
4Efficient-CapsNet/digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4Efficient-CapsNet/digit_caps/map/strided_slice/stack║
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_1║
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6Efficient-CapsNet/digit_caps/map/strided_slice/stack_2и
.Efficient-CapsNet/digit_caps/map/strided_sliceStridedSlice/Efficient-CapsNet/digit_caps/map/Shape:output:0=Efficient-CapsNet/digit_caps/map/strided_slice/stack:output:0?Efficient-CapsNet/digit_caps/map/strided_slice/stack_1:output:0?Efficient-CapsNet/digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.Efficient-CapsNet/digit_caps/map/strided_slice╟
<Efficient-CapsNet/digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2>
<Efficient-CapsNet/digit_caps/map/TensorArrayV2/element_shape┤
.Efficient-CapsNet/digit_caps/map/TensorArrayV2TensorListReserveEEfficient-CapsNet/digit_caps/map/TensorArrayV2/element_shape:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.Efficient-CapsNet/digit_caps/map/TensorArrayV2Й
VEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2X
VEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeД
HEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor6Efficient-CapsNet/digit_caps/digit_cap_inputs:output:0_Efficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02J
HEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensorТ
&Efficient-CapsNet/digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2(
&Efficient-CapsNet/digit_caps/map/Const╦
>Efficient-CapsNet/digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2@
>Efficient-CapsNet/digit_caps/map/TensorArrayV2_1/element_shape║
0Efficient-CapsNet/digit_caps/map/TensorArrayV2_1TensorListReserveGEfficient-CapsNet/digit_caps/map/TensorArrayV2_1/element_shape:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0Efficient-CapsNet/digit_caps/map/TensorArrayV2_1м
3Efficient-CapsNet/digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 25
3Efficient-CapsNet/digit_caps/map/while/loop_counter╘
&Efficient-CapsNet/digit_caps/map/whileWhile<Efficient-CapsNet/digit_caps/map/while/loop_counter:output:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0/Efficient-CapsNet/digit_caps/map/Const:output:09Efficient-CapsNet/digit_caps/map/TensorArrayV2_1:handle:07Efficient-CapsNet/digit_caps/map/strided_slice:output:0XEfficient-CapsNet/digit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0.efficient_capsnet_digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *<
body4R2
0Efficient-CapsNet_digit_caps_map_while_body_6632*<
cond4R2
0Efficient-CapsNet_digit_caps_map_while_cond_6631*!
output_shapes
: : : : : : : 2(
&Efficient-CapsNet/digit_caps/map/while 
QEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2S
QEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeЇ
CEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStack/Efficient-CapsNet/digit_caps/map/while:output:3ZEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:         
*
element_dtype02E
CEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStackЫ
2Efficient-CapsNet/digit_caps/digit_cap_predictionsSqueezeLEfficient-CapsNet/digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:         
*
squeeze_dims

         24
2Efficient-CapsNet/digit_caps/digit_cap_predictions╕
1Efficient-CapsNet/digit_caps/digit_cap_attentionsBatchMatMulV2;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
*
adj_y(23
1Efficient-CapsNet/digit_caps/digit_cap_attentionsх
 Efficient-CapsNet/digit_caps/mulMul"efficient_capsnet_digit_caps_mul_x:Efficient-CapsNet/digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:         
2"
 Efficient-CapsNet/digit_caps/mul│
2Efficient-CapsNet/digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        24
2Efficient-CapsNet/digit_caps/Sum/reduction_indices∙
 Efficient-CapsNet/digit_caps/SumSum$Efficient-CapsNet/digit_caps/mul:z:0;Efficient-CapsNet/digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:         
*
	keep_dims(2"
 Efficient-CapsNet/digit_caps/SumИ
!Efficient-CapsNet/digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :2#
!Efficient-CapsNet/digit_caps/RankУ
"Efficient-CapsNet/digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
■        2$
"Efficient-CapsNet/digit_caps/add/x╟
 Efficient-CapsNet/digit_caps/addAddV2+Efficient-CapsNet/digit_caps/add/x:output:0*Efficient-CapsNet/digit_caps/Rank:output:0*
T0*
_output_shapes
: 2"
 Efficient-CapsNet/digit_caps/addМ
#Efficient-CapsNet/digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2%
#Efficient-CapsNet/digit_caps/Rank_1К
"Efficient-CapsNet/digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"Efficient-CapsNet/digit_caps/mod/y─
 Efficient-CapsNet/digit_caps/modFloorMod$Efficient-CapsNet/digit_caps/add:z:0+Efficient-CapsNet/digit_caps/mod/y:output:0*
T0*
_output_shapes
: 2"
 Efficient-CapsNet/digit_caps/modК
"Efficient-CapsNet/digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"Efficient-CapsNet/digit_caps/Sub/y╟
 Efficient-CapsNet/digit_caps/SubSub,Efficient-CapsNet/digit_caps/Rank_1:output:0+Efficient-CapsNet/digit_caps/Sub/y:output:0*
T0*
_output_shapes
: 2"
 Efficient-CapsNet/digit_caps/SubЦ
(Efficient-CapsNet/digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(Efficient-CapsNet/digit_caps/range/startЦ
(Efficient-CapsNet/digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(Efficient-CapsNet/digit_caps/range/delta∙
"Efficient-CapsNet/digit_caps/rangeRange1Efficient-CapsNet/digit_caps/range/start:output:0$Efficient-CapsNet/digit_caps/mod:z:01Efficient-CapsNet/digit_caps/range/delta:output:0*
_output_shapes
:2$
"Efficient-CapsNet/digit_caps/rangeО
$Efficient-CapsNet/digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$Efficient-CapsNet/digit_caps/add_1/y╟
"Efficient-CapsNet/digit_caps/add_1AddV2$Efficient-CapsNet/digit_caps/mod:z:0-Efficient-CapsNet/digit_caps/add_1/y:output:0*
T0*
_output_shapes
: 2$
"Efficient-CapsNet/digit_caps/add_1Ъ
*Efficient-CapsNet/digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*Efficient-CapsNet/digit_caps/range_1/deltaЄ
$Efficient-CapsNet/digit_caps/range_1Range&Efficient-CapsNet/digit_caps/add_1:z:0$Efficient-CapsNet/digit_caps/Sub:z:03Efficient-CapsNet/digit_caps/range_1/delta:output:0*
_output_shapes
: 2&
$Efficient-CapsNet/digit_caps/range_1╕
,Efficient-CapsNet/digit_caps/concat/values_1Pack$Efficient-CapsNet/digit_caps/Sub:z:0*
N*
T0*
_output_shapes
:2.
,Efficient-CapsNet/digit_caps/concat/values_1╕
,Efficient-CapsNet/digit_caps/concat/values_3Pack$Efficient-CapsNet/digit_caps/mod:z:0*
N*
T0*
_output_shapes
:2.
,Efficient-CapsNet/digit_caps/concat/values_3Ц
(Efficient-CapsNet/digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(Efficient-CapsNet/digit_caps/concat/axisБ
#Efficient-CapsNet/digit_caps/concatConcatV2+Efficient-CapsNet/digit_caps/range:output:05Efficient-CapsNet/digit_caps/concat/values_1:output:0-Efficient-CapsNet/digit_caps/range_1:output:05Efficient-CapsNet/digit_caps/concat/values_3:output:01Efficient-CapsNet/digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#Efficient-CapsNet/digit_caps/concatЁ
&Efficient-CapsNet/digit_caps/transpose	Transpose)Efficient-CapsNet/digit_caps/Sum:output:0,Efficient-CapsNet/digit_caps/concat:output:0*
T0*/
_output_shapes
:         
2(
&Efficient-CapsNet/digit_caps/transpose╜
$Efficient-CapsNet/digit_caps/SoftmaxSoftmax*Efficient-CapsNet/digit_caps/transpose:y:0*
T0*/
_output_shapes
:         
2&
$Efficient-CapsNet/digit_caps/SoftmaxО
$Efficient-CapsNet/digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$Efficient-CapsNet/digit_caps/Sub_1/y═
"Efficient-CapsNet/digit_caps/Sub_1Sub,Efficient-CapsNet/digit_caps/Rank_1:output:0-Efficient-CapsNet/digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: 2$
"Efficient-CapsNet/digit_caps/Sub_1Ъ
*Efficient-CapsNet/digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Efficient-CapsNet/digit_caps/range_2/startЪ
*Efficient-CapsNet/digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*Efficient-CapsNet/digit_caps/range_2/deltaБ
$Efficient-CapsNet/digit_caps/range_2Range3Efficient-CapsNet/digit_caps/range_2/start:output:0$Efficient-CapsNet/digit_caps/mod:z:03Efficient-CapsNet/digit_caps/range_2/delta:output:0*
_output_shapes
:2&
$Efficient-CapsNet/digit_caps/range_2О
$Efficient-CapsNet/digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2&
$Efficient-CapsNet/digit_caps/add_2/y╟
"Efficient-CapsNet/digit_caps/add_2AddV2$Efficient-CapsNet/digit_caps/mod:z:0-Efficient-CapsNet/digit_caps/add_2/y:output:0*
T0*
_output_shapes
: 2$
"Efficient-CapsNet/digit_caps/add_2Ъ
*Efficient-CapsNet/digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2,
*Efficient-CapsNet/digit_caps/range_3/deltaЇ
$Efficient-CapsNet/digit_caps/range_3Range&Efficient-CapsNet/digit_caps/add_2:z:0&Efficient-CapsNet/digit_caps/Sub_1:z:03Efficient-CapsNet/digit_caps/range_3/delta:output:0*
_output_shapes
: 2&
$Efficient-CapsNet/digit_caps/range_3╛
.Efficient-CapsNet/digit_caps/concat_1/values_1Pack&Efficient-CapsNet/digit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:20
.Efficient-CapsNet/digit_caps/concat_1/values_1╝
.Efficient-CapsNet/digit_caps/concat_1/values_3Pack$Efficient-CapsNet/digit_caps/mod:z:0*
N*
T0*
_output_shapes
:20
.Efficient-CapsNet/digit_caps/concat_1/values_3Ъ
*Efficient-CapsNet/digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*Efficient-CapsNet/digit_caps/concat_1/axisН
%Efficient-CapsNet/digit_caps/concat_1ConcatV2-Efficient-CapsNet/digit_caps/range_2:output:07Efficient-CapsNet/digit_caps/concat_1/values_1:output:0-Efficient-CapsNet/digit_caps/range_3:output:07Efficient-CapsNet/digit_caps/concat_1/values_3:output:03Efficient-CapsNet/digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%Efficient-CapsNet/digit_caps/concat_1г
<Efficient-CapsNet/digit_caps/digit_cap_coupling_coefficients	Transpose.Efficient-CapsNet/digit_caps/Softmax:softmax:0.Efficient-CapsNet/digit_caps/concat_1:output:0*
T0*/
_output_shapes
:         
2>
<Efficient-CapsNet/digit_caps/digit_cap_coupling_coefficientsх
1Efficient-CapsNet/digit_caps/add_3/ReadVariableOpReadVariableOp:efficient_capsnet_digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype023
1Efficient-CapsNet/digit_caps/add_3/ReadVariableOpИ
"Efficient-CapsNet/digit_caps/add_3AddV2@Efficient-CapsNet/digit_caps/digit_cap_coupling_coefficients:y:09Efficient-CapsNet/digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2$
"Efficient-CapsNet/digit_caps/add_3·
#Efficient-CapsNet/digit_caps/MatMulBatchMatMulV2&Efficient-CapsNet/digit_caps/add_3:z:0;Efficient-CapsNet/digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
2%
#Efficient-CapsNet/digit_caps/MatMul█
$Efficient-CapsNet/digit_caps/SqueezeSqueeze,Efficient-CapsNet/digit_caps/MatMul:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

■        2&
$Efficient-CapsNet/digit_caps/SqueezeЛ
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/mulMul-Efficient-CapsNet/digit_caps/Squeeze:output:0-Efficient-CapsNet/digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:         
28
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/mulч
HEfficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2J
HEfficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum/reduction_indices═
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/SumSum:Efficient-CapsNet/digit_caps/digit_cap_squash/norm/mul:z:0QEfficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(28
6Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sumё
7Efficient-CapsNet/digit_caps/digit_cap_squash/norm/SqrtSqrt?Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         
29
7Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sqrtр
1Efficient-CapsNet/digit_caps/digit_cap_squash/ExpExp;Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         
23
1Efficient-CapsNet/digit_caps/digit_cap_squash/Exp╖
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv/xи
5Efficient-CapsNet/digit_caps/digit_cap_squash/truedivRealDiv@Efficient-CapsNet/digit_caps/digit_cap_squash/truediv/x:output:05Efficient-CapsNet/digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         
27
5Efficient-CapsNet/digit_caps/digit_cap_squash/truedivп
3Efficient-CapsNet/digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?25
3Efficient-CapsNet/digit_caps/digit_cap_squash/sub/xЬ
1Efficient-CapsNet/digit_caps/digit_cap_squash/subSub<Efficient-CapsNet/digit_caps/digit_cap_squash/sub/x:output:09Efficient-CapsNet/digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         
23
1Efficient-CapsNet/digit_caps/digit_cap_squash/subп
3Efficient-CapsNet/digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓325
3Efficient-CapsNet/digit_caps/digit_cap_squash/add/yа
1Efficient-CapsNet/digit_caps/digit_cap_squash/addAddV2;Efficient-CapsNet/digit_caps/digit_cap_squash/norm/Sqrt:y:0<Efficient-CapsNet/digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         
23
1Efficient-CapsNet/digit_caps/digit_cap_squash/addЩ
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv_1RealDiv-Efficient-CapsNet/digit_caps/Squeeze:output:05Efficient-CapsNet/digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:         
29
7Efficient-CapsNet/digit_caps/digit_cap_squash/truediv_1Ч
1Efficient-CapsNet/digit_caps/digit_cap_squash/mulMul5Efficient-CapsNet/digit_caps/digit_cap_squash/sub:z:0;Efficient-CapsNet/digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         
23
1Efficient-CapsNet/digit_caps/digit_cap_squash/mul√
&Efficient-CapsNet/digit_probs/norm/mulMul5Efficient-CapsNet/digit_caps/digit_cap_squash/mul:z:05Efficient-CapsNet/digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:         
2(
&Efficient-CapsNet/digit_probs/norm/mul╟
8Efficient-CapsNet/digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2:
8Efficient-CapsNet/digit_probs/norm/Sum/reduction_indicesН
&Efficient-CapsNet/digit_probs/norm/SumSum*Efficient-CapsNet/digit_probs/norm/mul:z:0AEfficient-CapsNet/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2(
&Efficient-CapsNet/digit_probs/norm/Sum┴
'Efficient-CapsNet/digit_probs/norm/SqrtSqrt/Efficient-CapsNet/digit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:         
2)
'Efficient-CapsNet/digit_probs/norm/Sqrtт
*Efficient-CapsNet/digit_probs/norm/SqueezeSqueeze+Efficient-CapsNet/digit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2,
*Efficient-CapsNet/digit_probs/norm/SqueezeО
IdentityIdentity3Efficient-CapsNet/digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity┬
NoOpNoOp2^Efficient-CapsNet/digit_caps/add_3/ReadVariableOp'^Efficient-CapsNet/digit_caps/map/whileH^Efficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpH^Efficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpG^Efficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpQ^Efficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1Q^Efficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpS^Efficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@^Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOpB^Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1H^Efficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpG^Efficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2f
1Efficient-CapsNet/digit_caps/add_3/ReadVariableOp1Efficient-CapsNet/digit_caps/add_3/ReadVariableOp2P
&Efficient-CapsNet/digit_caps/map/while&Efficient-CapsNet/digit_caps/map/while2Т
GEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2Р
FEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2Т
GEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2Р
FEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2Т
GEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2Р
FEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2Т
GEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpGEfficient-CapsNet/feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2Р
FEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOpFEfficient-CapsNet/feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2д
PEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2и
REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12В
?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp2Ж
AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm1/ReadVariableOp_12д
PEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2и
REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12В
?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp2Ж
AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm2/ReadVariableOp_12д
PEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2и
REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12В
?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp2Ж
AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm3/ReadVariableOp_12д
PEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpPEfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2и
REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1REfficient-CapsNet/feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12В
?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp?Efficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp2Ж
AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_1AEfficient-CapsNet/feature_maps/feature_map_norm4/ReadVariableOp_12Т
GEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpGEfficient-CapsNet/primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2Р
FEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpFEfficient-CapsNet/primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
ц
р
0Efficient-CapsNet_digit_caps_map_while_cond_6631^
Zefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterY
Uefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice6
2efficient_capsnet_digit_caps_map_while_placeholder8
4efficient_capsnet_digit_caps_map_while_placeholder_1^
Zefficient_capsnet_digit_caps_map_while_less_efficient_capsnet_digit_caps_map_strided_slicet
pefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_cond_6631___redundant_placeholder0t
pefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_cond_6631___redundant_placeholder13
/efficient_capsnet_digit_caps_map_while_identity
У
+Efficient-CapsNet/digit_caps/map/while/LessLess2efficient_capsnet_digit_caps_map_while_placeholderZefficient_capsnet_digit_caps_map_while_less_efficient_capsnet_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2-
+Efficient-CapsNet/digit_caps/map/while/Less║
-Efficient-CapsNet/digit_caps/map/while/Less_1LessZefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_while_loop_counterUefficient_capsnet_digit_caps_map_while_efficient_capsnet_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2/
-Efficient-CapsNet/digit_caps/map/while/Less_1Ё
1Efficient-CapsNet/digit_caps/map/while/LogicalAnd
LogicalAnd1Efficient-CapsNet/digit_caps/map/while/Less_1:z:0/Efficient-CapsNet/digit_caps/map/while/Less:z:0*
_output_shapes
: 23
1Efficient-CapsNet/digit_caps/map/while/LogicalAnd╞
/Efficient-CapsNet/digit_caps/map/while/IdentityIdentity5Efficient-CapsNet/digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 21
/Efficient-CapsNet/digit_caps/map/while/Identity"k
/efficient_capsnet_digit_caps_map_while_identity8Efficient-CapsNet/digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
В
ф
digit_caps_map_while_cond_8593:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_1:
6digit_caps_map_while_less_digit_caps_map_strided_sliceP
Ldigit_caps_map_while_digit_caps_map_while_cond_8593___redundant_placeholder0P
Ldigit_caps_map_while_digit_caps_map_while_cond_8593___redundant_placeholder1!
digit_caps_map_while_identity
╣
digit_caps/map/while/LessLess digit_caps_map_while_placeholder6digit_caps_map_while_less_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2
digit_caps/map/while/Less╬
digit_caps/map/while/Less_1Less6digit_caps_map_while_digit_caps_map_while_loop_counter1digit_caps_map_while_digit_caps_map_strided_slice*
T0*
_output_shapes
: 2
digit_caps/map/while/Less_1и
digit_caps/map/while/LogicalAnd
LogicalAnddigit_caps/map/while/Less_1:z:0digit_caps/map/while/Less:z:0*
_output_shapes
: 2!
digit_caps/map/while/LogicalAndР
digit_caps/map/while/IdentityIdentity#digit_caps/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
digit_caps/map/while/Identity"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
э!
т	
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_7577

inputs+
feature_maps_7342: 
feature_maps_7344: 
feature_maps_7346: 
feature_maps_7348: 
feature_maps_7350: 
feature_maps_7352: +
feature_maps_7354: @
feature_maps_7356:@
feature_maps_7358:@
feature_maps_7360:@
feature_maps_7362:@
feature_maps_7364:@+
feature_maps_7366:@@
feature_maps_7368:@
feature_maps_7370:@
feature_maps_7372:@
feature_maps_7374:@
feature_maps_7376:@,
feature_maps_7378:@А 
feature_maps_7380:	А 
feature_maps_7382:	А 
feature_maps_7384:	А 
feature_maps_7386:	А 
feature_maps_7388:	А,
primary_caps_7425:		А 
primary_caps_7427:	А)
digit_caps_7558:

digit_caps_7560%
digit_caps_7562:

identityИв"digit_caps/StatefulPartitionedCallв$feature_maps/StatefulPartitionedCallв$primary_caps/StatefulPartitionedCall№
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_7342feature_maps_7344feature_maps_7346feature_maps_7348feature_maps_7350feature_maps_7352feature_maps_7354feature_maps_7356feature_maps_7358feature_maps_7360feature_maps_7362feature_maps_7364feature_maps_7366feature_maps_7368feature_maps_7370feature_maps_7372feature_maps_7374feature_maps_7376feature_maps_7378feature_maps_7380feature_maps_7382feature_maps_7384feature_maps_7386feature_maps_7388*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_73412&
$feature_maps/StatefulPartitionedCall╨
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_7425primary_caps_7427*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_primary_caps_layer_call_and_return_conditional_losses_74242&
$primary_caps/StatefulPartitionedCall╪
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_7558digit_caps_7560digit_caps_7562*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_digit_caps_layer_call_and_return_conditional_losses_75572$
"digit_caps/StatefulPartitionedCallГ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_75742
digit_probs/PartitionedCall
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity┴
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
▌
Ц
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9545

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
э
Ъ
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9669

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
щ
Г
+__inference_feature_maps_layer_call_fn_9002
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А
identityИвStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_73412
StatefulPartitionedCallД
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:         		А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
х!
т	
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8006

inputs+
feature_maps_7943: 
feature_maps_7945: 
feature_maps_7947: 
feature_maps_7949: 
feature_maps_7951: 
feature_maps_7953: +
feature_maps_7955: @
feature_maps_7957:@
feature_maps_7959:@
feature_maps_7961:@
feature_maps_7963:@
feature_maps_7965:@+
feature_maps_7967:@@
feature_maps_7969:@
feature_maps_7971:@
feature_maps_7973:@
feature_maps_7975:@
feature_maps_7977:@,
feature_maps_7979:@А 
feature_maps_7981:	А 
feature_maps_7983:	А 
feature_maps_7985:	А 
feature_maps_7987:	А 
feature_maps_7989:	А,
primary_caps_7992:		А 
primary_caps_7994:	А)
digit_caps_7997:

digit_caps_7999%
digit_caps_8001:

identityИв"digit_caps/StatefulPartitionedCallв$feature_maps/StatefulPartitionedCallв$primary_caps/StatefulPartitionedCallЇ
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinputsfeature_maps_7943feature_maps_7945feature_maps_7947feature_maps_7949feature_maps_7951feature_maps_7953feature_maps_7955feature_maps_7957feature_maps_7959feature_maps_7961feature_maps_7963feature_maps_7965feature_maps_7967feature_maps_7969feature_maps_7971feature_maps_7973feature_maps_7975feature_maps_7977feature_maps_7979feature_maps_7981feature_maps_7983feature_maps_7985feature_maps_7987feature_maps_7989*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_78242&
$feature_maps/StatefulPartitionedCall╨
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_7992primary_caps_7994*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_primary_caps_layer_call_and_return_conditional_losses_74242&
$primary_caps/StatefulPartitionedCall╪
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_7997digit_caps_7999digit_caps_8001*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_digit_caps_layer_call_and_return_conditional_losses_75572$
"digit_caps/StatefulPartitionedCallГ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_76552
digit_probs/PartitionedCall
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity┴
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
ў!
ш	
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8262
input_images+
feature_maps_8199: 
feature_maps_8201: 
feature_maps_8203: 
feature_maps_8205: 
feature_maps_8207: 
feature_maps_8209: +
feature_maps_8211: @
feature_maps_8213:@
feature_maps_8215:@
feature_maps_8217:@
feature_maps_8219:@
feature_maps_8221:@+
feature_maps_8223:@@
feature_maps_8225:@
feature_maps_8227:@
feature_maps_8229:@
feature_maps_8231:@
feature_maps_8233:@,
feature_maps_8235:@А 
feature_maps_8237:	А 
feature_maps_8239:	А 
feature_maps_8241:	А 
feature_maps_8243:	А 
feature_maps_8245:	А,
primary_caps_8248:		А 
primary_caps_8250:	А)
digit_caps_8253:

digit_caps_8255%
digit_caps_8257:

identityИв"digit_caps/StatefulPartitionedCallв$feature_maps/StatefulPartitionedCallв$primary_caps/StatefulPartitionedCall·
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_8199feature_maps_8201feature_maps_8203feature_maps_8205feature_maps_8207feature_maps_8209feature_maps_8211feature_maps_8213feature_maps_8215feature_maps_8217feature_maps_8219feature_maps_8221feature_maps_8223feature_maps_8225feature_maps_8227feature_maps_8229feature_maps_8231feature_maps_8233feature_maps_8235feature_maps_8237feature_maps_8239feature_maps_8241feature_maps_8243feature_maps_8245*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_78242&
$feature_maps/StatefulPartitionedCall╨
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_8248primary_caps_8250*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_primary_caps_layer_call_and_return_conditional_losses_74242&
$primary_caps/StatefulPartitionedCall╪
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_8253digit_caps_8255digit_caps_8257*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_digit_caps_layer_call_and_return_conditional_losses_75572$
"digit_caps/StatefulPartitionedCallГ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_76552
digit_probs/PartitionedCall
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity┴
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
╜	
╧
0__inference_feature_map_norm4_layer_call_fn_9651

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИвStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,                           А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_71862
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
▌
Ц
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9483

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Г'
г
digit_caps_map_while_body_8839:
6digit_caps_map_while_digit_caps_map_while_loop_counter5
1digit_caps_map_while_digit_caps_map_strided_slice$
 digit_caps_map_while_placeholder&
"digit_caps_map_while_placeholder_19
5digit_caps_map_while_digit_caps_map_strided_slice_1_0u
qdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0O
5digit_caps_map_while_matmul_readvariableop_resource_0:
!
digit_caps_map_while_identity#
digit_caps_map_while_identity_1#
digit_caps_map_while_identity_2#
digit_caps_map_while_identity_37
3digit_caps_map_while_digit_caps_map_strided_slice_1s
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorM
3digit_caps_map_while_matmul_readvariableop_resource:
Ив*digit_caps/map/while/MatMul/ReadVariableOpщ
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2H
Fdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeм
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0 digit_caps_map_while_placeholderOdigit_caps/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype02:
8digit_caps/map/while/TensorArrayV2Read/TensorListGetItem╓
*digit_caps/map/while/MatMul/ReadVariableOpReadVariableOp5digit_caps_map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype02,
*digit_caps/map/while/MatMul/ReadVariableOpё
digit_caps/map/while/MatMulBatchMatMulV22digit_caps/map/while/MatMul/ReadVariableOp:value:0?digit_caps/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
2
digit_caps/map/while/MatMulд
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"digit_caps_map_while_placeholder_1 digit_caps_map_while_placeholder$digit_caps/map/while/MatMul:output:0*
_output_shapes
: *
element_dtype02;
9digit_caps/map/while/TensorArrayV2Write/TensorListSetItemz
digit_caps/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/map/while/add/yе
digit_caps/map/while/addAddV2 digit_caps_map_while_placeholder#digit_caps/map/while/add/y:output:0*
T0*
_output_shapes
: 2
digit_caps/map/while/add~
digit_caps/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/map/while/add_1/y┴
digit_caps/map/while/add_1AddV26digit_caps_map_while_digit_caps_map_while_loop_counter%digit_caps/map/while/add_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/map/while/add_1з
digit_caps/map/while/IdentityIdentitydigit_caps/map/while/add_1:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2
digit_caps/map/while/Identity╛
digit_caps/map/while/Identity_1Identity1digit_caps_map_while_digit_caps_map_strided_slice^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_1й
digit_caps/map/while/Identity_2Identitydigit_caps/map/while/add:z:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_2╓
digit_caps/map/while/Identity_3IdentityIdigit_caps/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^digit_caps/map/while/NoOp*
T0*
_output_shapes
: 2!
digit_caps/map/while/Identity_3е
digit_caps/map/while/NoOpNoOp+^digit_caps/map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
digit_caps/map/while/NoOp"l
3digit_caps_map_while_digit_caps_map_strided_slice_15digit_caps_map_while_digit_caps_map_strided_slice_1_0"G
digit_caps_map_while_identity&digit_caps/map/while/Identity:output:0"K
digit_caps_map_while_identity_1(digit_caps/map/while/Identity_1:output:0"K
digit_caps_map_while_identity_2(digit_caps/map/while/Identity_2:output:0"K
digit_caps_map_while_identity_3(digit_caps/map/while/Identity_3:output:0"l
3digit_caps_map_while_matmul_readvariableop_resource5digit_caps_map_while_matmul_readvariableop_resource_0"ф
odigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensorqdigit_caps_map_while_tensorarrayv2read_tensorlistgetitem_digit_caps_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2X
*digit_caps/map/while/MatMul/ReadVariableOp*digit_caps/map/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
─с
┐!
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8949

inputsW
=feature_maps_feature_map_conv1_conv2d_readvariableop_resource: L
>feature_maps_feature_map_conv1_biasadd_readvariableop_resource: D
6feature_maps_feature_map_norm1_readvariableop_resource: F
8feature_maps_feature_map_norm1_readvariableop_1_resource: U
Gfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: W
Ifeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: W
=feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @L
>feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm2_readvariableop_resource:@F
8feature_maps_feature_map_norm2_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@W
=feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@L
>feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm3_readvariableop_resource:@F
8feature_maps_feature_map_norm3_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@X
=feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@АM
>feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	АE
6feature_maps_feature_map_norm4_readvariableop_resource:	АG
8feature_maps_feature_map_norm4_readvariableop_1_resource:	АV
Gfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АX
Ifeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	АX
=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		АM
>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	А6
digit_caps_map_while_input_6:

digit_caps_mul_x>
(digit_caps_add_3_readvariableop_resource:

identityИвdigit_caps/add_3/ReadVariableOpвdigit_caps/map/whileв5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpв-feature_maps/feature_map_norm1/AssignNewValueв/feature_maps/feature_map_norm1/AssignNewValue_1в>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm1/ReadVariableOpв/feature_maps/feature_map_norm1/ReadVariableOp_1в-feature_maps/feature_map_norm2/AssignNewValueв/feature_maps/feature_map_norm2/AssignNewValue_1в>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm2/ReadVariableOpв/feature_maps/feature_map_norm2/ReadVariableOp_1в-feature_maps/feature_map_norm3/AssignNewValueв/feature_maps/feature_map_norm3/AssignNewValue_1в>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm3/ReadVariableOpв/feature_maps/feature_map_norm3/ReadVariableOp_1в-feature_maps/feature_map_norm4/AssignNewValueв/feature_maps/feature_map_norm4/AssignNewValue_1в>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm4/ReadVariableOpв/feature_maps/feature_map_norm4/ReadVariableOp_1в5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpв4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpЄ
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpБ
%feature_maps/feature_map_conv1/Conv2DConv2Dinputs<feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2'
%feature_maps/feature_map_conv1/Conv2Dщ
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv1/BiasAddBiasAdd.feature_maps/feature_map_conv1/Conv2D:output:0=feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2(
&feature_maps/feature_map_conv1/BiasAdd╜
#feature_maps/feature_map_conv1/ReluRelu/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2%
#feature_maps/feature_map_conv1/Relu╤
-feature_maps/feature_map_norm1/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02/
-feature_maps/feature_map_norm1/ReadVariableOp╫
/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype021
/feature_maps/feature_map_norm1/ReadVariableOp_1Д
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1╜
/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv1/Relu:activations:05feature_maps/feature_map_norm1/ReadVariableOp:value:07feature_maps/feature_map_norm1/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<21
/feature_maps/feature_map_norm1/FusedBatchNormV3▌
-feature_maps/feature_map_norm1/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm1/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-feature_maps/feature_map_norm1/AssignNewValueщ
/feature_maps/feature_map_norm1/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm1/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/feature_maps/feature_map_norm1/AssignNewValue_1Є
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype026
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpо
%feature_maps/feature_map_conv2/Conv2DConv2D3feature_maps/feature_map_norm1/FusedBatchNormV3:y:0<feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv2/Conv2Dщ
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv2/BiasAddBiasAdd.feature_maps/feature_map_conv2/Conv2D:output:0=feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2(
&feature_maps/feature_map_conv2/BiasAdd╜
#feature_maps/feature_map_conv2/ReluRelu/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2%
#feature_maps/feature_map_conv2/Relu╤
-feature_maps/feature_map_norm2/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02/
-feature_maps/feature_map_norm2/ReadVariableOp╫
/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/feature_maps/feature_map_norm2/ReadVariableOp_1Д
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1╜
/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv2/Relu:activations:05feature_maps/feature_map_norm2/ReadVariableOp:value:07feature_maps/feature_map_norm2/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<21
/feature_maps/feature_map_norm2/FusedBatchNormV3▌
-feature_maps/feature_map_norm2/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm2/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-feature_maps/feature_map_norm2/AssignNewValueщ
/feature_maps/feature_map_norm2/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm2/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/feature_maps/feature_map_norm2/AssignNewValue_1Є
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype026
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpо
%feature_maps/feature_map_conv3/Conv2DConv2D3feature_maps/feature_map_norm2/FusedBatchNormV3:y:0<feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv3/Conv2Dщ
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv3/BiasAddBiasAdd.feature_maps/feature_map_conv3/Conv2D:output:0=feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2(
&feature_maps/feature_map_conv3/BiasAdd╜
#feature_maps/feature_map_conv3/ReluRelu/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2%
#feature_maps/feature_map_conv3/Relu╤
-feature_maps/feature_map_norm3/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02/
-feature_maps/feature_map_norm3/ReadVariableOp╫
/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/feature_maps/feature_map_norm3/ReadVariableOp_1Д
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1╜
/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv3/Relu:activations:05feature_maps/feature_map_norm3/ReadVariableOp:value:07feature_maps/feature_map_norm3/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<21
/feature_maps/feature_map_norm3/FusedBatchNormV3▌
-feature_maps/feature_map_norm3/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm3/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-feature_maps/feature_map_norm3/AssignNewValueщ
/feature_maps/feature_map_norm3/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm3/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/feature_maps/feature_map_norm3/AssignNewValue_1є
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype026
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpп
%feature_maps/feature_map_conv4/Conv2DConv2D3feature_maps/feature_map_norm3/FusedBatchNormV3:y:0<feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv4/Conv2Dъ
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype027
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpЕ
&feature_maps/feature_map_conv4/BiasAddBiasAdd.feature_maps/feature_map_conv4/Conv2D:output:0=feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2(
&feature_maps/feature_map_conv4/BiasAdd╛
#feature_maps/feature_map_conv4/ReluRelu/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2%
#feature_maps/feature_map_conv4/Relu╥
-feature_maps/feature_map_norm4/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-feature_maps/feature_map_norm4/ReadVariableOp╪
/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/feature_maps/feature_map_norm4/ReadVariableOp_1Е
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpЛ
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1┬
/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv4/Relu:activations:05feature_maps/feature_map_norm4/ReadVariableOp:value:07feature_maps/feature_map_norm4/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<21
/feature_maps/feature_map_norm4/FusedBatchNormV3▌
-feature_maps/feature_map_norm4/AssignNewValueAssignVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource<feature_maps/feature_map_norm4/FusedBatchNormV3:batch_mean:0?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-feature_maps/feature_map_norm4/AssignNewValueщ
/feature_maps/feature_map_norm4/AssignNewValue_1AssignVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource@feature_maps/feature_map_norm4/FusedBatchNormV3:batch_variance:0A^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/feature_maps/feature_map_norm4/AssignNewValue_1є
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		А*
dtype026
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpп
%primary_caps/primary_cap_dconv/Conv2DConv2D3feature_maps/feature_map_norm4/FusedBatchNormV3:y:0<primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2'
%primary_caps/primary_cap_dconv/Conv2Dъ
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype027
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpЕ
&primary_caps/primary_cap_dconv/BiasAddBiasAdd.primary_caps/primary_cap_dconv/Conv2D:output:0=primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2(
&primary_caps/primary_cap_dconv/BiasAdd╛
#primary_caps/primary_cap_dconv/ReluRelu/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:         А2%
#primary_caps/primary_cap_dconv/Relu▒
&primary_caps/primary_cap_reshape/ShapeShape1primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:2(
&primary_caps/primary_cap_reshape/Shape╢
4primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4primary_caps/primary_cap_reshape/strided_slice/stack║
6primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6primary_caps/primary_cap_reshape/strided_slice/stack_1║
6primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6primary_caps/primary_cap_reshape/strided_slice/stack_2и
.primary_caps/primary_cap_reshape/strided_sliceStridedSlice/primary_caps/primary_cap_reshape/Shape:output:0=primary_caps/primary_cap_reshape/strided_slice/stack:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.primary_caps/primary_cap_reshape/strided_sliceп
0primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         22
0primary_caps/primary_cap_reshape/Reshape/shape/1ж
0primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0primary_caps/primary_cap_reshape/Reshape/shape/2┼
.primary_caps/primary_cap_reshape/Reshape/shapePack7primary_caps/primary_cap_reshape/strided_slice:output:09primary_caps/primary_cap_reshape/Reshape/shape/1:output:09primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:20
.primary_caps/primary_cap_reshape/Reshape/shapeБ
(primary_caps/primary_cap_reshape/ReshapeReshape1primary_caps/primary_cap_dconv/Relu:activations:07primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2*
(primary_caps/primary_cap_reshape/Reshapeў
(primary_caps/primary_cap_squash/norm/mulMul1primary_caps/primary_cap_reshape/Reshape:output:01primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:         2*
(primary_caps/primary_cap_squash/norm/mul╦
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2<
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesХ
(primary_caps/primary_cap_squash/norm/SumSum,primary_caps/primary_cap_squash/norm/mul:z:0Cprimary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         *
	keep_dims(2*
(primary_caps/primary_cap_squash/norm/Sum╟
)primary_caps/primary_cap_squash/norm/SqrtSqrt1primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         2+
)primary_caps/primary_cap_squash/norm/Sqrt╢
#primary_caps/primary_cap_squash/ExpExp-primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/ExpЫ
)primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)primary_caps/primary_cap_squash/truediv/xЁ
'primary_caps/primary_cap_squash/truedivRealDiv2primary_caps/primary_cap_squash/truediv/x:output:0'primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         2)
'primary_caps/primary_cap_squash/truedivУ
%primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%primary_caps/primary_cap_squash/sub/xф
#primary_caps/primary_cap_squash/subSub.primary_caps/primary_cap_squash/sub/x:output:0+primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/subУ
%primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32'
%primary_caps/primary_cap_squash/add/yш
#primary_caps/primary_cap_squash/addAddV2-primary_caps/primary_cap_squash/norm/Sqrt:y:0.primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/addє
)primary_caps/primary_cap_squash/truediv_1RealDiv1primary_caps/primary_cap_reshape/Reshape:output:0'primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:         2+
)primary_caps/primary_cap_squash/truediv_1▀
#primary_caps/primary_cap_squash/mulMul'primary_caps/primary_cap_squash/sub:z:0-primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/mulx
digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/ExpandDims/dim├
digit_caps/ExpandDims
ExpandDims'primary_caps/primary_cap_squash/mul:z:0"digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
digit_caps/ExpandDimsП
digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         2
digit_caps/Tile/multiplesи
digit_caps/TileTiledigit_caps/ExpandDims:output:0"digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:         
2
digit_caps/TileН
digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2!
digit_caps/digit_cap_inputs/dim╩
digit_caps/digit_cap_inputs
ExpandDimsdigit_caps/Tile:output:0(digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:         
2
digit_caps/digit_cap_inputsА
digit_caps/map/ShapeShape$digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:2
digit_caps/map/ShapeТ
"digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"digit_caps/map/strided_slice/stackЦ
$digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$digit_caps/map/strided_slice/stack_1Ц
$digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$digit_caps/map/strided_slice/stack_2╝
digit_caps/map/strided_sliceStridedSlicedigit_caps/map/Shape:output:0+digit_caps/map/strided_slice/stack:output:0-digit_caps/map/strided_slice/stack_1:output:0-digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
digit_caps/map/strided_sliceг
*digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*digit_caps/map/TensorArrayV2/element_shapeь
digit_caps/map/TensorArrayV2TensorListReserve3digit_caps/map/TensorArrayV2/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
digit_caps/map/TensorArrayV2х
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2F
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape╝
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$digit_caps/digit_cap_inputs:output:0Mdigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorn
digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/map/Constз
,digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,digit_caps/map/TensorArrayV2_1/element_shapeЄ
digit_caps/map/TensorArrayV2_1TensorListReserve5digit_caps/map/TensorArrayV2_1/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
digit_caps/map/TensorArrayV2_1И
!digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!digit_caps/map/while/loop_counterО
digit_caps/map/whileWhile*digit_caps/map/while/loop_counter:output:0%digit_caps/map/strided_slice:output:0digit_caps/map/Const:output:0'digit_caps/map/TensorArrayV2_1:handle:0%digit_caps/map/strided_slice:output:0Fdigit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( **
body"R 
digit_caps_map_while_body_8839**
cond"R 
digit_caps_map_while_cond_8838*!
output_shapes
: : : : : : : 2
digit_caps/map/while█
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2A
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeм
1digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStackdigit_caps/map/while:output:3Hdigit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:         
*
element_dtype023
1digit_caps/map/TensorArrayV2Stack/TensorListStackх
 digit_caps/digit_cap_predictionsSqueeze:digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:         
*
squeeze_dims

         2"
 digit_caps/digit_cap_predictionsЁ
digit_caps/digit_cap_attentionsBatchMatMulV2)digit_caps/digit_cap_predictions:output:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
*
adj_y(2!
digit_caps/digit_cap_attentionsЭ
digit_caps/mulMuldigit_caps_mul_x(digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:         
2
digit_caps/mulП
 digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        2"
 digit_caps/Sum/reduction_indices▒
digit_caps/SumSumdigit_caps/mul:z:0)digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:         
*
	keep_dims(2
digit_caps/Sumd
digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Ranko
digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
■        2
digit_caps/add/x
digit_caps/addAddV2digit_caps/add/x:output:0digit_caps/Rank:output:0*
T0*
_output_shapes
: 2
digit_caps/addh
digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Rank_1f
digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/mod/y|
digit_caps/modFloorModdigit_caps/add:z:0digit_caps/mod/y:output:0*
T0*
_output_shapes
: 2
digit_caps/modf
digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Sub/y
digit_caps/SubSubdigit_caps/Rank_1:output:0digit_caps/Sub/y:output:0*
T0*
_output_shapes
: 2
digit_caps/Subr
digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/range/startr
digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range/deltaЯ
digit_caps/rangeRangedigit_caps/range/start:output:0digit_caps/mod:z:0digit_caps/range/delta:output:0*
_output_shapes
:2
digit_caps/rangej
digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/add_1/y
digit_caps/add_1AddV2digit_caps/mod:z:0digit_caps/add_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/add_1v
digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_1/deltaШ
digit_caps/range_1Rangedigit_caps/add_1:z:0digit_caps/Sub:z:0!digit_caps/range_1/delta:output:0*
_output_shapes
: 2
digit_caps/range_1В
digit_caps/concat/values_1Packdigit_caps/Sub:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat/values_1В
digit_caps/concat/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat/values_3r
digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/concat/axisГ
digit_caps/concatConcatV2digit_caps/range:output:0#digit_caps/concat/values_1:output:0digit_caps/range_1:output:0#digit_caps/concat/values_3:output:0digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:2
digit_caps/concatи
digit_caps/transpose	Transposedigit_caps/Sum:output:0digit_caps/concat:output:0*
T0*/
_output_shapes
:         
2
digit_caps/transposeЗ
digit_caps/SoftmaxSoftmaxdigit_caps/transpose:y:0*
T0*/
_output_shapes
:         
2
digit_caps/Softmaxj
digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Sub_1/yЕ
digit_caps/Sub_1Subdigit_caps/Rank_1:output:0digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/Sub_1v
digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/range_2/startv
digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_2/deltaз
digit_caps/range_2Range!digit_caps/range_2/start:output:0digit_caps/mod:z:0!digit_caps/range_2/delta:output:0*
_output_shapes
:2
digit_caps/range_2j
digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/add_2/y
digit_caps/add_2AddV2digit_caps/mod:z:0digit_caps/add_2/y:output:0*
T0*
_output_shapes
: 2
digit_caps/add_2v
digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_3/deltaЪ
digit_caps/range_3Rangedigit_caps/add_2:z:0digit_caps/Sub_1:z:0!digit_caps/range_3/delta:output:0*
_output_shapes
: 2
digit_caps/range_3И
digit_caps/concat_1/values_1Packdigit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1/values_1Ж
digit_caps/concat_1/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1/values_3v
digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/concat_1/axisП
digit_caps/concat_1ConcatV2digit_caps/range_2:output:0%digit_caps/concat_1/values_1:output:0digit_caps/range_3:output:0%digit_caps/concat_1/values_3:output:0!digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1█
*digit_caps/digit_cap_coupling_coefficients	Transposedigit_caps/Softmax:softmax:0digit_caps/concat_1:output:0*
T0*/
_output_shapes
:         
2,
*digit_caps/digit_cap_coupling_coefficientsп
digit_caps/add_3/ReadVariableOpReadVariableOp(digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype02!
digit_caps/add_3/ReadVariableOp└
digit_caps/add_3AddV2.digit_caps/digit_cap_coupling_coefficients:y:0'digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
digit_caps/add_3▓
digit_caps/MatMulBatchMatMulV2digit_caps/add_3:z:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
2
digit_caps/MatMulе
digit_caps/SqueezeSqueezedigit_caps/MatMul:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

■        2
digit_caps/Squeeze├
$digit_caps/digit_cap_squash/norm/mulMuldigit_caps/Squeeze:output:0digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:         
2&
$digit_caps/digit_cap_squash/norm/mul├
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         28
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesЕ
$digit_caps/digit_cap_squash/norm/SumSum(digit_caps/digit_cap_squash/norm/mul:z:0?digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2&
$digit_caps/digit_cap_squash/norm/Sum╗
%digit_caps/digit_cap_squash/norm/SqrtSqrt-digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         
2'
%digit_caps/digit_cap_squash/norm/Sqrtк
digit_caps/digit_cap_squash/ExpExp)digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/ExpУ
%digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%digit_caps/digit_cap_squash/truediv/xр
#digit_caps/digit_cap_squash/truedivRealDiv.digit_caps/digit_cap_squash/truediv/x:output:0#digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         
2%
#digit_caps/digit_cap_squash/truedivЛ
!digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!digit_caps/digit_cap_squash/sub/x╘
digit_caps/digit_cap_squash/subSub*digit_caps/digit_cap_squash/sub/x:output:0'digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/subЛ
!digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32#
!digit_caps/digit_cap_squash/add/y╪
digit_caps/digit_cap_squash/addAddV2)digit_caps/digit_cap_squash/norm/Sqrt:y:0*digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/add╤
%digit_caps/digit_cap_squash/truediv_1RealDivdigit_caps/Squeeze:output:0#digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:         
2'
%digit_caps/digit_cap_squash/truediv_1╧
digit_caps/digit_cap_squash/mulMul#digit_caps/digit_cap_squash/sub:z:0)digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/mul│
digit_probs/norm/mulMul#digit_caps/digit_cap_squash/mul:z:0#digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:         
2
digit_probs/norm/mulг
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&digit_probs/norm/Sum/reduction_indices┼
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2
digit_probs/norm/SumЛ
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:         
2
digit_probs/norm/Sqrtм
digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
digit_probs/norm/Squeeze|
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity╥
NoOpNoOp ^digit_caps/add_3/ReadVariableOp^digit_caps/map/while6^feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv1/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv2/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv3/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv4/Conv2D/ReadVariableOp.^feature_maps/feature_map_norm1/AssignNewValue0^feature_maps/feature_map_norm1/AssignNewValue_1?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm1/ReadVariableOp0^feature_maps/feature_map_norm1/ReadVariableOp_1.^feature_maps/feature_map_norm2/AssignNewValue0^feature_maps/feature_map_norm2/AssignNewValue_1?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm2/ReadVariableOp0^feature_maps/feature_map_norm2/ReadVariableOp_1.^feature_maps/feature_map_norm3/AssignNewValue0^feature_maps/feature_map_norm3/AssignNewValue_1?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm3/ReadVariableOp0^feature_maps/feature_map_norm3/ReadVariableOp_1.^feature_maps/feature_map_norm4/AssignNewValue0^feature_maps/feature_map_norm4/AssignNewValue_1?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm4/ReadVariableOp0^feature_maps/feature_map_norm4/ReadVariableOp_16^primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5^primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
digit_caps/add_3/ReadVariableOpdigit_caps/add_3/ReadVariableOp2,
digit_caps/map/whiledigit_caps/map/while2n
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2^
-feature_maps/feature_map_norm1/AssignNewValue-feature_maps/feature_map_norm1/AssignNewValue2b
/feature_maps/feature_map_norm1/AssignNewValue_1/feature_maps/feature_map_norm1/AssignNewValue_12А
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm1/ReadVariableOp-feature_maps/feature_map_norm1/ReadVariableOp2b
/feature_maps/feature_map_norm1/ReadVariableOp_1/feature_maps/feature_map_norm1/ReadVariableOp_12^
-feature_maps/feature_map_norm2/AssignNewValue-feature_maps/feature_map_norm2/AssignNewValue2b
/feature_maps/feature_map_norm2/AssignNewValue_1/feature_maps/feature_map_norm2/AssignNewValue_12А
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm2/ReadVariableOp-feature_maps/feature_map_norm2/ReadVariableOp2b
/feature_maps/feature_map_norm2/ReadVariableOp_1/feature_maps/feature_map_norm2/ReadVariableOp_12^
-feature_maps/feature_map_norm3/AssignNewValue-feature_maps/feature_map_norm3/AssignNewValue2b
/feature_maps/feature_map_norm3/AssignNewValue_1/feature_maps/feature_map_norm3/AssignNewValue_12А
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm3/ReadVariableOp-feature_maps/feature_map_norm3/ReadVariableOp2b
/feature_maps/feature_map_norm3/ReadVariableOp_1/feature_maps/feature_map_norm3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/AssignNewValue-feature_maps/feature_map_norm4/AssignNewValue2b
/feature_maps/feature_map_norm4/AssignNewValue_1/feature_maps/feature_map_norm4/AssignNewValue_12А
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/ReadVariableOp-feature_maps/feature_map_norm4/ReadVariableOp2b
/feature_maps/feature_map_norm4/ReadVariableOp_1/feature_maps/feature_map_norm4/ReadVariableOp_12n
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2l
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
н
Ф
"__inference_signature_wrapper_8333
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А%

unknown_23:		А

unknown_24:	А$

unknown_25:


unknown_26 

unknown_27:

identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__wrapped_model_67422
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
╖	
╦
0__inference_feature_map_norm1_layer_call_fn_9452

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_67642
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╔
F
*__inference_digit_probs_layer_call_fn_9416

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_75742
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
ы
▒
map_while_body_9306$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0D
*map_while_matmul_readvariableop_resource_0:

map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorB
(map_while_matmul_readvariableop_resource:
Ивmap/while/MatMul/ReadVariableOp╙
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeъ
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItem╡
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype02!
map/while/MatMul/ReadVariableOp┼
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
2
map/while/MatMulэ
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/yК
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1{
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/IdentityЗ
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_1}
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_2к
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_3Д
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
map/while/NoOp"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"╕
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2B
map/while/MatMul/ReadVariableOpmap/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╡	
╦
0__inference_feature_map_norm1_layer_call_fn_9465

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_68082
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
э
Ъ
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_7142

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1с
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
ы
▒
map_while_body_7452$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0D
*map_while_matmul_readvariableop_resource_0:

map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensorB
(map_while_matmul_readvariableop_resource:
Ивmap/while/MatMul/ReadVariableOp╙
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeъ
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*&
_output_shapes
:
*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItem╡
map/while/MatMul/ReadVariableOpReadVariableOp*map_while_matmul_readvariableop_resource_0*&
_output_shapes
:
*
dtype02!
map/while/MatMul/ReadVariableOp┼
map/while/MatMulBatchMatMulV2'map/while/MatMul/ReadVariableOp:value:04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*&
_output_shapes
:
2
map/while/MatMulэ
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholdermap/while/MatMul:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/yК
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1{
map/while/IdentityIdentitymap/while/add_1:z:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/IdentityЗ
map/while/Identity_1Identitymap_while_map_strided_slice^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_1}
map/while/Identity_2Identitymap/while/add:z:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_2к
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^map/while/NoOp*
T0*
_output_shapes
: 2
map/while/Identity_3Д
map/while/NoOpNoOp ^map/while/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
map/while/NoOp"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"V
(map_while_matmul_readvariableop_resource*map_while_matmul_readvariableop_resource_0"╕
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : : : 2B
map/while/MatMul/ReadVariableOpmap/while/MatMul/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▀
в
0__inference_Efficient-CapsNet_layer_call_fn_8130
input_images!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5: @
	unknown_6:@
	unknown_7:@
	unknown_8:@
	unknown_9:@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А%

unknown_23:		А

unknown_24:	А$

unknown_25:


unknown_26 

unknown_27:

identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinput_imagesunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_80062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
╝╛
╖
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8704

inputsW
=feature_maps_feature_map_conv1_conv2d_readvariableop_resource: L
>feature_maps_feature_map_conv1_biasadd_readvariableop_resource: D
6feature_maps_feature_map_norm1_readvariableop_resource: F
8feature_maps_feature_map_norm1_readvariableop_1_resource: U
Gfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource: W
Ifeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: W
=feature_maps_feature_map_conv2_conv2d_readvariableop_resource: @L
>feature_maps_feature_map_conv2_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm2_readvariableop_resource:@F
8feature_maps_feature_map_norm2_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@W
=feature_maps_feature_map_conv3_conv2d_readvariableop_resource:@@L
>feature_maps_feature_map_conv3_biasadd_readvariableop_resource:@D
6feature_maps_feature_map_norm3_readvariableop_resource:@F
8feature_maps_feature_map_norm3_readvariableop_1_resource:@U
Gfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@W
Ifeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@X
=feature_maps_feature_map_conv4_conv2d_readvariableop_resource:@АM
>feature_maps_feature_map_conv4_biasadd_readvariableop_resource:	АE
6feature_maps_feature_map_norm4_readvariableop_resource:	АG
8feature_maps_feature_map_norm4_readvariableop_1_resource:	АV
Gfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АX
Ifeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	АX
=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource:		АM
>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource:	А6
digit_caps_map_while_input_6:

digit_caps_mul_x>
(digit_caps_add_3_readvariableop_resource:

identityИвdigit_caps/add_3/ReadVariableOpвdigit_caps/map/whileв5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpв5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpв4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpв>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm1/ReadVariableOpв/feature_maps/feature_map_norm1/ReadVariableOp_1в>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm2/ReadVariableOpв/feature_maps/feature_map_norm2/ReadVariableOp_1в>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm3/ReadVariableOpв/feature_maps/feature_map_norm3/ReadVariableOp_1в>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpв@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в-feature_maps/feature_map_norm4/ReadVariableOpв/feature_maps/feature_map_norm4/ReadVariableOp_1в5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpв4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpЄ
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOpБ
%feature_maps/feature_map_conv1/Conv2DConv2Dinputs<feature_maps/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2'
%feature_maps/feature_map_conv1/Conv2Dщ
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv1/BiasAddBiasAdd.feature_maps/feature_map_conv1/Conv2D:output:0=feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2(
&feature_maps/feature_map_conv1/BiasAdd╜
#feature_maps/feature_map_conv1/ReluRelu/feature_maps/feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2%
#feature_maps/feature_map_conv1/Relu╤
-feature_maps/feature_map_norm1/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02/
-feature_maps/feature_map_norm1/ReadVariableOp╫
/feature_maps/feature_map_norm1/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype021
/feature_maps/feature_map_norm1/ReadVariableOp_1Д
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1п
/feature_maps/feature_map_norm1/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv1/Relu:activations:05feature_maps/feature_map_norm1/ReadVariableOp:value:07feature_maps/feature_map_norm1/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 21
/feature_maps/feature_map_norm1/FusedBatchNormV3Є
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype026
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOpо
%feature_maps/feature_map_conv2/Conv2DConv2D3feature_maps/feature_map_norm1/FusedBatchNormV3:y:0<feature_maps/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv2/Conv2Dщ
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv2/BiasAddBiasAdd.feature_maps/feature_map_conv2/Conv2D:output:0=feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2(
&feature_maps/feature_map_conv2/BiasAdd╜
#feature_maps/feature_map_conv2/ReluRelu/feature_maps/feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2%
#feature_maps/feature_map_conv2/Relu╤
-feature_maps/feature_map_norm2/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02/
-feature_maps/feature_map_norm2/ReadVariableOp╫
/feature_maps/feature_map_norm2/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/feature_maps/feature_map_norm2/ReadVariableOp_1Д
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1п
/feature_maps/feature_map_norm2/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv2/Relu:activations:05feature_maps/feature_map_norm2/ReadVariableOp:value:07feature_maps/feature_map_norm2/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 21
/feature_maps/feature_map_norm2/FusedBatchNormV3Є
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype026
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOpо
%feature_maps/feature_map_conv3/Conv2DConv2D3feature_maps/feature_map_norm2/FusedBatchNormV3:y:0<feature_maps/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv3/Conv2Dщ
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype027
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOpД
&feature_maps/feature_map_conv3/BiasAddBiasAdd.feature_maps/feature_map_conv3/Conv2D:output:0=feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2(
&feature_maps/feature_map_conv3/BiasAdd╜
#feature_maps/feature_map_conv3/ReluRelu/feature_maps/feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2%
#feature_maps/feature_map_conv3/Relu╤
-feature_maps/feature_map_norm3/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02/
-feature_maps/feature_map_norm3/ReadVariableOp╫
/feature_maps/feature_map_norm3/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype021
/feature_maps/feature_map_norm3/ReadVariableOp_1Д
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpК
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1п
/feature_maps/feature_map_norm3/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv3/Relu:activations:05feature_maps/feature_map_norm3/ReadVariableOp:value:07feature_maps/feature_map_norm3/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 21
/feature_maps/feature_map_norm3/FusedBatchNormV3є
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp=feature_maps_feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype026
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOpп
%feature_maps/feature_map_conv4/Conv2DConv2D3feature_maps/feature_map_norm3/FusedBatchNormV3:y:0<feature_maps/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2'
%feature_maps/feature_map_conv4/Conv2Dъ
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp>feature_maps_feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype027
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOpЕ
&feature_maps/feature_map_conv4/BiasAddBiasAdd.feature_maps/feature_map_conv4/Conv2D:output:0=feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2(
&feature_maps/feature_map_conv4/BiasAdd╛
#feature_maps/feature_map_conv4/ReluRelu/feature_maps/feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2%
#feature_maps/feature_map_conv4/Relu╥
-feature_maps/feature_map_norm4/ReadVariableOpReadVariableOp6feature_maps_feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02/
-feature_maps/feature_map_norm4/ReadVariableOp╪
/feature_maps/feature_map_norm4/ReadVariableOp_1ReadVariableOp8feature_maps_feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype021
/feature_maps/feature_map_norm4/ReadVariableOp_1Е
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOpGfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02@
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpЛ
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIfeature_maps_feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02B
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1┤
/feature_maps/feature_map_norm4/FusedBatchNormV3FusedBatchNormV31feature_maps/feature_map_conv4/Relu:activations:05feature_maps/feature_map_norm4/ReadVariableOp:value:07feature_maps/feature_map_norm4/ReadVariableOp_1:value:0Ffeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0Hfeature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
is_training( 21
/feature_maps/feature_map_norm4/FusedBatchNormV3є
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpReadVariableOp=primary_caps_primary_cap_dconv_conv2d_readvariableop_resource*'
_output_shapes
:		А*
dtype026
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOpп
%primary_caps/primary_cap_dconv/Conv2DConv2D3feature_maps/feature_map_norm4/FusedBatchNormV3:y:0<primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А*
paddingVALID*
strides
2'
%primary_caps/primary_cap_dconv/Conv2Dъ
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpReadVariableOp>primary_caps_primary_cap_dconv_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype027
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOpЕ
&primary_caps/primary_cap_dconv/BiasAddBiasAdd.primary_caps/primary_cap_dconv/Conv2D:output:0=primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         А2(
&primary_caps/primary_cap_dconv/BiasAdd╛
#primary_caps/primary_cap_dconv/ReluRelu/primary_caps/primary_cap_dconv/BiasAdd:output:0*
T0*0
_output_shapes
:         А2%
#primary_caps/primary_cap_dconv/Relu▒
&primary_caps/primary_cap_reshape/ShapeShape1primary_caps/primary_cap_dconv/Relu:activations:0*
T0*
_output_shapes
:2(
&primary_caps/primary_cap_reshape/Shape╢
4primary_caps/primary_cap_reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4primary_caps/primary_cap_reshape/strided_slice/stack║
6primary_caps/primary_cap_reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6primary_caps/primary_cap_reshape/strided_slice/stack_1║
6primary_caps/primary_cap_reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6primary_caps/primary_cap_reshape/strided_slice/stack_2и
.primary_caps/primary_cap_reshape/strided_sliceStridedSlice/primary_caps/primary_cap_reshape/Shape:output:0=primary_caps/primary_cap_reshape/strided_slice/stack:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_1:output:0?primary_caps/primary_cap_reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.primary_caps/primary_cap_reshape/strided_sliceп
0primary_caps/primary_cap_reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
         22
0primary_caps/primary_cap_reshape/Reshape/shape/1ж
0primary_caps/primary_cap_reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0primary_caps/primary_cap_reshape/Reshape/shape/2┼
.primary_caps/primary_cap_reshape/Reshape/shapePack7primary_caps/primary_cap_reshape/strided_slice:output:09primary_caps/primary_cap_reshape/Reshape/shape/1:output:09primary_caps/primary_cap_reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:20
.primary_caps/primary_cap_reshape/Reshape/shapeБ
(primary_caps/primary_cap_reshape/ReshapeReshape1primary_caps/primary_cap_dconv/Relu:activations:07primary_caps/primary_cap_reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         2*
(primary_caps/primary_cap_reshape/Reshapeў
(primary_caps/primary_cap_squash/norm/mulMul1primary_caps/primary_cap_reshape/Reshape:output:01primary_caps/primary_cap_reshape/Reshape:output:0*
T0*+
_output_shapes
:         2*
(primary_caps/primary_cap_squash/norm/mul╦
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2<
:primary_caps/primary_cap_squash/norm/Sum/reduction_indicesХ
(primary_caps/primary_cap_squash/norm/SumSum,primary_caps/primary_cap_squash/norm/mul:z:0Cprimary_caps/primary_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         *
	keep_dims(2*
(primary_caps/primary_cap_squash/norm/Sum╟
)primary_caps/primary_cap_squash/norm/SqrtSqrt1primary_caps/primary_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         2+
)primary_caps/primary_cap_squash/norm/Sqrt╢
#primary_caps/primary_cap_squash/ExpExp-primary_caps/primary_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/ExpЫ
)primary_caps/primary_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)primary_caps/primary_cap_squash/truediv/xЁ
'primary_caps/primary_cap_squash/truedivRealDiv2primary_caps/primary_cap_squash/truediv/x:output:0'primary_caps/primary_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         2)
'primary_caps/primary_cap_squash/truedivУ
%primary_caps/primary_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%primary_caps/primary_cap_squash/sub/xф
#primary_caps/primary_cap_squash/subSub.primary_caps/primary_cap_squash/sub/x:output:0+primary_caps/primary_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/subУ
%primary_caps/primary_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32'
%primary_caps/primary_cap_squash/add/yш
#primary_caps/primary_cap_squash/addAddV2-primary_caps/primary_cap_squash/norm/Sqrt:y:0.primary_caps/primary_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/addє
)primary_caps/primary_cap_squash/truediv_1RealDiv1primary_caps/primary_cap_reshape/Reshape:output:0'primary_caps/primary_cap_squash/add:z:0*
T0*+
_output_shapes
:         2+
)primary_caps/primary_cap_squash/truediv_1▀
#primary_caps/primary_cap_squash/mulMul'primary_caps/primary_cap_squash/sub:z:0-primary_caps/primary_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         2%
#primary_caps/primary_cap_squash/mulx
digit_caps/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/ExpandDims/dim├
digit_caps/ExpandDims
ExpandDims'primary_caps/primary_cap_squash/mul:z:0"digit_caps/ExpandDims/dim:output:0*
T0*/
_output_shapes
:         2
digit_caps/ExpandDimsП
digit_caps/Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         2
digit_caps/Tile/multiplesи
digit_caps/TileTiledigit_caps/ExpandDims:output:0"digit_caps/Tile/multiples:output:0*
T0*/
_output_shapes
:         
2
digit_caps/TileН
digit_caps/digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2!
digit_caps/digit_cap_inputs/dim╩
digit_caps/digit_cap_inputs
ExpandDimsdigit_caps/Tile:output:0(digit_caps/digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:         
2
digit_caps/digit_cap_inputsА
digit_caps/map/ShapeShape$digit_caps/digit_cap_inputs:output:0*
T0*
_output_shapes
:2
digit_caps/map/ShapeТ
"digit_caps/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"digit_caps/map/strided_slice/stackЦ
$digit_caps/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$digit_caps/map/strided_slice/stack_1Ц
$digit_caps/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$digit_caps/map/strided_slice/stack_2╝
digit_caps/map/strided_sliceStridedSlicedigit_caps/map/Shape:output:0+digit_caps/map/strided_slice/stack:output:0-digit_caps/map/strided_slice/stack_1:output:0-digit_caps/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
digit_caps/map/strided_sliceг
*digit_caps/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*digit_caps/map/TensorArrayV2/element_shapeь
digit_caps/map/TensorArrayV2TensorListReserve3digit_caps/map/TensorArrayV2/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
digit_caps/map/TensorArrayV2х
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2F
Ddigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape╝
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor$digit_caps/digit_cap_inputs:output:0Mdigit_caps/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6digit_caps/map/TensorArrayUnstack/TensorListFromTensorn
digit_caps/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/map/Constз
,digit_caps/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2.
,digit_caps/map/TensorArrayV2_1/element_shapeЄ
digit_caps/map/TensorArrayV2_1TensorListReserve5digit_caps/map/TensorArrayV2_1/element_shape:output:0%digit_caps/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
digit_caps/map/TensorArrayV2_1И
!digit_caps/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!digit_caps/map/while/loop_counterО
digit_caps/map/whileWhile*digit_caps/map/while/loop_counter:output:0%digit_caps/map/strided_slice:output:0digit_caps/map/Const:output:0'digit_caps/map/TensorArrayV2_1:handle:0%digit_caps/map/strided_slice:output:0Fdigit_caps/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0digit_caps_map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( **
body"R 
digit_caps_map_while_body_8594**
cond"R 
digit_caps_map_while_cond_8593*!
output_shapes
: : : : : : : 2
digit_caps/map/while█
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2A
?digit_caps/map/TensorArrayV2Stack/TensorListStack/element_shapeм
1digit_caps/map/TensorArrayV2Stack/TensorListStackTensorListStackdigit_caps/map/while:output:3Hdigit_caps/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:         
*
element_dtype023
1digit_caps/map/TensorArrayV2Stack/TensorListStackх
 digit_caps/digit_cap_predictionsSqueeze:digit_caps/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:         
*
squeeze_dims

         2"
 digit_caps/digit_cap_predictionsЁ
digit_caps/digit_cap_attentionsBatchMatMulV2)digit_caps/digit_cap_predictions:output:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
*
adj_y(2!
digit_caps/digit_cap_attentionsЭ
digit_caps/mulMuldigit_caps_mul_x(digit_caps/digit_cap_attentions:output:0*
T0*/
_output_shapes
:         
2
digit_caps/mulП
 digit_caps/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        2"
 digit_caps/Sum/reduction_indices▒
digit_caps/SumSumdigit_caps/mul:z:0)digit_caps/Sum/reduction_indices:output:0*
T0*/
_output_shapes
:         
*
	keep_dims(2
digit_caps/Sumd
digit_caps/RankConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Ranko
digit_caps/add/xConst*
_output_shapes
: *
dtype0*
valueB :
■        2
digit_caps/add/x
digit_caps/addAddV2digit_caps/add/x:output:0digit_caps/Rank:output:0*
T0*
_output_shapes
: 2
digit_caps/addh
digit_caps/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Rank_1f
digit_caps/mod/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/mod/y|
digit_caps/modFloorModdigit_caps/add:z:0digit_caps/mod/y:output:0*
T0*
_output_shapes
: 2
digit_caps/modf
digit_caps/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Sub/y
digit_caps/SubSubdigit_caps/Rank_1:output:0digit_caps/Sub/y:output:0*
T0*
_output_shapes
: 2
digit_caps/Subr
digit_caps/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/range/startr
digit_caps/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range/deltaЯ
digit_caps/rangeRangedigit_caps/range/start:output:0digit_caps/mod:z:0digit_caps/range/delta:output:0*
_output_shapes
:2
digit_caps/rangej
digit_caps/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/add_1/y
digit_caps/add_1AddV2digit_caps/mod:z:0digit_caps/add_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/add_1v
digit_caps/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_1/deltaШ
digit_caps/range_1Rangedigit_caps/add_1:z:0digit_caps/Sub:z:0!digit_caps/range_1/delta:output:0*
_output_shapes
: 2
digit_caps/range_1В
digit_caps/concat/values_1Packdigit_caps/Sub:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat/values_1В
digit_caps/concat/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat/values_3r
digit_caps/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/concat/axisГ
digit_caps/concatConcatV2digit_caps/range:output:0#digit_caps/concat/values_1:output:0digit_caps/range_1:output:0#digit_caps/concat/values_3:output:0digit_caps/concat/axis:output:0*
N*
T0*
_output_shapes
:2
digit_caps/concatи
digit_caps/transpose	Transposedigit_caps/Sum:output:0digit_caps/concat:output:0*
T0*/
_output_shapes
:         
2
digit_caps/transposeЗ
digit_caps/SoftmaxSoftmaxdigit_caps/transpose:y:0*
T0*/
_output_shapes
:         
2
digit_caps/Softmaxj
digit_caps/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/Sub_1/yЕ
digit_caps/Sub_1Subdigit_caps/Rank_1:output:0digit_caps/Sub_1/y:output:0*
T0*
_output_shapes
: 2
digit_caps/Sub_1v
digit_caps/range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/range_2/startv
digit_caps/range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_2/deltaз
digit_caps/range_2Range!digit_caps/range_2/start:output:0digit_caps/mod:z:0!digit_caps/range_2/delta:output:0*
_output_shapes
:2
digit_caps/range_2j
digit_caps/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/add_2/y
digit_caps/add_2AddV2digit_caps/mod:z:0digit_caps/add_2/y:output:0*
T0*
_output_shapes
: 2
digit_caps/add_2v
digit_caps/range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
digit_caps/range_3/deltaЪ
digit_caps/range_3Rangedigit_caps/add_2:z:0digit_caps/Sub_1:z:0!digit_caps/range_3/delta:output:0*
_output_shapes
: 2
digit_caps/range_3И
digit_caps/concat_1/values_1Packdigit_caps/Sub_1:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1/values_1Ж
digit_caps/concat_1/values_3Packdigit_caps/mod:z:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1/values_3v
digit_caps/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
digit_caps/concat_1/axisП
digit_caps/concat_1ConcatV2digit_caps/range_2:output:0%digit_caps/concat_1/values_1:output:0digit_caps/range_3:output:0%digit_caps/concat_1/values_3:output:0!digit_caps/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
digit_caps/concat_1█
*digit_caps/digit_cap_coupling_coefficients	Transposedigit_caps/Softmax:softmax:0digit_caps/concat_1:output:0*
T0*/
_output_shapes
:         
2,
*digit_caps/digit_cap_coupling_coefficientsп
digit_caps/add_3/ReadVariableOpReadVariableOp(digit_caps_add_3_readvariableop_resource*"
_output_shapes
:
*
dtype02!
digit_caps/add_3/ReadVariableOp└
digit_caps/add_3AddV2.digit_caps/digit_cap_coupling_coefficients:y:0'digit_caps/add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
digit_caps/add_3▓
digit_caps/MatMulBatchMatMulV2digit_caps/add_3:z:0)digit_caps/digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
2
digit_caps/MatMulе
digit_caps/SqueezeSqueezedigit_caps/MatMul:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

■        2
digit_caps/Squeeze├
$digit_caps/digit_cap_squash/norm/mulMuldigit_caps/Squeeze:output:0digit_caps/Squeeze:output:0*
T0*+
_output_shapes
:         
2&
$digit_caps/digit_cap_squash/norm/mul├
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         28
6digit_caps/digit_cap_squash/norm/Sum/reduction_indicesЕ
$digit_caps/digit_cap_squash/norm/SumSum(digit_caps/digit_cap_squash/norm/mul:z:0?digit_caps/digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2&
$digit_caps/digit_cap_squash/norm/Sum╗
%digit_caps/digit_cap_squash/norm/SqrtSqrt-digit_caps/digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         
2'
%digit_caps/digit_cap_squash/norm/Sqrtк
digit_caps/digit_cap_squash/ExpExp)digit_caps/digit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/ExpУ
%digit_caps/digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2'
%digit_caps/digit_cap_squash/truediv/xр
#digit_caps/digit_cap_squash/truedivRealDiv.digit_caps/digit_cap_squash/truediv/x:output:0#digit_caps/digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         
2%
#digit_caps/digit_cap_squash/truedivЛ
!digit_caps/digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2#
!digit_caps/digit_cap_squash/sub/x╘
digit_caps/digit_cap_squash/subSub*digit_caps/digit_cap_squash/sub/x:output:0'digit_caps/digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/subЛ
!digit_caps/digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32#
!digit_caps/digit_cap_squash/add/y╪
digit_caps/digit_cap_squash/addAddV2)digit_caps/digit_cap_squash/norm/Sqrt:y:0*digit_caps/digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/add╤
%digit_caps/digit_cap_squash/truediv_1RealDivdigit_caps/Squeeze:output:0#digit_caps/digit_cap_squash/add:z:0*
T0*+
_output_shapes
:         
2'
%digit_caps/digit_cap_squash/truediv_1╧
digit_caps/digit_cap_squash/mulMul#digit_caps/digit_cap_squash/sub:z:0)digit_caps/digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         
2!
digit_caps/digit_cap_squash/mul│
digit_probs/norm/mulMul#digit_caps/digit_cap_squash/mul:z:0#digit_caps/digit_cap_squash/mul:z:0*
T0*+
_output_shapes
:         
2
digit_probs/norm/mulг
&digit_probs/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2(
&digit_probs/norm/Sum/reduction_indices┼
digit_probs/norm/SumSumdigit_probs/norm/mul:z:0/digit_probs/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2
digit_probs/norm/SumЛ
digit_probs/norm/SqrtSqrtdigit_probs/norm/Sum:output:0*
T0*+
_output_shapes
:         
2
digit_probs/norm/Sqrtм
digit_probs/norm/SqueezeSqueezedigit_probs/norm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
digit_probs/norm/Squeeze|
IdentityIdentity!digit_probs/norm/Squeeze:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity╩
NoOpNoOp ^digit_caps/add_3/ReadVariableOp^digit_caps/map/while6^feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv1/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv2/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv3/Conv2D/ReadVariableOp6^feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5^feature_maps/feature_map_conv4/Conv2D/ReadVariableOp?^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm1/ReadVariableOp0^feature_maps/feature_map_norm1/ReadVariableOp_1?^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm2/ReadVariableOp0^feature_maps/feature_map_norm2/ReadVariableOp_1?^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm3/ReadVariableOp0^feature_maps/feature_map_norm3/ReadVariableOp_1?^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOpA^feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1.^feature_maps/feature_map_norm4/ReadVariableOp0^feature_maps/feature_map_norm4/ReadVariableOp_16^primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5^primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2B
digit_caps/add_3/ReadVariableOpdigit_caps/add_3/ReadVariableOp2,
digit_caps/map/whiledigit_caps/map/while2n
5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv1/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp4feature_maps/feature_map_conv1/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv2/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp4feature_maps/feature_map_conv2/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv3/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp4feature_maps/feature_map_conv3/Conv2D/ReadVariableOp2n
5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp5feature_maps/feature_map_conv4/BiasAdd/ReadVariableOp2l
4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp4feature_maps/feature_map_conv4/Conv2D/ReadVariableOp2А
>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm1/ReadVariableOp-feature_maps/feature_map_norm1/ReadVariableOp2b
/feature_maps/feature_map_norm1/ReadVariableOp_1/feature_maps/feature_map_norm1/ReadVariableOp_12А
>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm2/ReadVariableOp-feature_maps/feature_map_norm2/ReadVariableOp2b
/feature_maps/feature_map_norm2/ReadVariableOp_1/feature_maps/feature_map_norm2/ReadVariableOp_12А
>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm3/ReadVariableOp-feature_maps/feature_map_norm3/ReadVariableOp2b
/feature_maps/feature_map_norm3/ReadVariableOp_1/feature_maps/feature_map_norm3/ReadVariableOp_12А
>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp>feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp2Д
@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1@feature_maps/feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12^
-feature_maps/feature_map_norm4/ReadVariableOp-feature_maps/feature_map_norm4/ReadVariableOp2b
/feature_maps/feature_map_norm4/ReadVariableOp_1/feature_maps/feature_map_norm4/ReadVariableOp_12n
5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp5primary_caps/primary_cap_dconv/BiasAdd/ReadVariableOp2l
4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp4primary_caps/primary_cap_dconv/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: 
Х
a
E__inference_digit_probs_layer_call_and_return_conditional_losses_9439

inputs
identitya
norm/mulMulinputsinputs*
T0*+
_output_shapes
:         
2

norm/mulЛ
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2
norm/Sum/reduction_indicesХ
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2

norm/Sumg
	norm/SqrtSqrtnorm/Sum:output:0*
T0*+
_output_shapes
:         
2
	norm/SqrtИ
norm/SqueezeSqueezenorm/Sqrt:y:0*
T0*'
_output_shapes
:         
*
squeeze_dims

         2
norm/Squeezei
IdentityIdentitynorm/Squeeze:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
 !
ш	
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8196
input_images+
feature_maps_8133: 
feature_maps_8135: 
feature_maps_8137: 
feature_maps_8139: 
feature_maps_8141: 
feature_maps_8143: +
feature_maps_8145: @
feature_maps_8147:@
feature_maps_8149:@
feature_maps_8151:@
feature_maps_8153:@
feature_maps_8155:@+
feature_maps_8157:@@
feature_maps_8159:@
feature_maps_8161:@
feature_maps_8163:@
feature_maps_8165:@
feature_maps_8167:@,
feature_maps_8169:@А 
feature_maps_8171:	А 
feature_maps_8173:	А 
feature_maps_8175:	А 
feature_maps_8177:	А 
feature_maps_8179:	А,
primary_caps_8182:		А 
primary_caps_8184:	А)
digit_caps_8187:

digit_caps_8189%
digit_caps_8191:

identityИв"digit_caps/StatefulPartitionedCallв$feature_maps/StatefulPartitionedCallв$primary_caps/StatefulPartitionedCallВ
$feature_maps/StatefulPartitionedCallStatefulPartitionedCallinput_imagesfeature_maps_8133feature_maps_8135feature_maps_8137feature_maps_8139feature_maps_8141feature_maps_8143feature_maps_8145feature_maps_8147feature_maps_8149feature_maps_8151feature_maps_8153feature_maps_8155feature_maps_8157feature_maps_8159feature_maps_8161feature_maps_8163feature_maps_8165feature_maps_8167feature_maps_8169feature_maps_8171feature_maps_8173feature_maps_8175feature_maps_8177feature_maps_8179*$
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:         		А*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_feature_maps_layer_call_and_return_conditional_losses_73412&
$feature_maps/StatefulPartitionedCall╨
$primary_caps/StatefulPartitionedCallStatefulPartitionedCall-feature_maps/StatefulPartitionedCall:output:0primary_caps_8182primary_caps_8184*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_primary_caps_layer_call_and_return_conditional_losses_74242&
$primary_caps/StatefulPartitionedCall╪
"digit_caps/StatefulPartitionedCallStatefulPartitionedCall-primary_caps/StatefulPartitionedCall:output:0digit_caps_8187digit_caps_8189digit_caps_8191*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_digit_caps_layer_call_and_return_conditional_losses_75572$
"digit_caps/StatefulPartitionedCallГ
digit_probs/PartitionedCallPartitionedCall+digit_caps/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_75742
digit_probs/PartitionedCall
IdentityIdentity$digit_probs/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         
2

Identity┴
NoOpNoOp#^digit_caps/StatefulPartitionedCall%^feature_maps/StatefulPartitionedCall%^primary_caps/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"digit_caps/StatefulPartitionedCall"digit_caps/StatefulPartitionedCall2L
$feature_maps/StatefulPartitionedCall$feature_maps/StatefulPartitionedCall2L
$primary_caps/StatefulPartitionedCall$primary_caps/StatefulPartitionedCall:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images:

_output_shapes
: 
К	
╩
map_while_cond_9305$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_9305___redundant_placeholder0:
6map_while_map_while_cond_9305___redundant_placeholder1
map_while_identity
В
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/LessМ
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Щ┌
у9
!__inference__traced_restore_10183
file_prefixQ
7assignvariableop_digit_caps_digit_caps_transform_tensor:
I
3assignvariableop_1_digit_caps_digit_caps_log_priors:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: R
8assignvariableop_7_feature_maps_feature_map_conv1_kernel: D
6assignvariableop_8_feature_maps_feature_map_conv1_bias: E
7assignvariableop_9_feature_maps_feature_map_norm1_gamma: E
7assignvariableop_10_feature_maps_feature_map_norm1_beta: S
9assignvariableop_11_feature_maps_feature_map_conv2_kernel: @E
7assignvariableop_12_feature_maps_feature_map_conv2_bias:@F
8assignvariableop_13_feature_maps_feature_map_norm2_gamma:@E
7assignvariableop_14_feature_maps_feature_map_norm2_beta:@S
9assignvariableop_15_feature_maps_feature_map_conv3_kernel:@@E
7assignvariableop_16_feature_maps_feature_map_conv3_bias:@F
8assignvariableop_17_feature_maps_feature_map_norm3_gamma:@E
7assignvariableop_18_feature_maps_feature_map_norm3_beta:@T
9assignvariableop_19_feature_maps_feature_map_conv4_kernel:@АF
7assignvariableop_20_feature_maps_feature_map_conv4_bias:	АG
8assignvariableop_21_feature_maps_feature_map_norm4_gamma:	АF
7assignvariableop_22_feature_maps_feature_map_norm4_beta:	АT
9assignvariableop_23_primary_caps_primary_cap_dconv_kernel:		АF
7assignvariableop_24_primary_caps_primary_cap_dconv_bias:	АL
>assignvariableop_25_feature_maps_feature_map_norm1_moving_mean: P
Bassignvariableop_26_feature_maps_feature_map_norm1_moving_variance: L
>assignvariableop_27_feature_maps_feature_map_norm2_moving_mean:@P
Bassignvariableop_28_feature_maps_feature_map_norm2_moving_variance:@L
>assignvariableop_29_feature_maps_feature_map_norm3_moving_mean:@P
Bassignvariableop_30_feature_maps_feature_map_norm3_moving_variance:@M
>assignvariableop_31_feature_maps_feature_map_norm4_moving_mean:	АQ
Bassignvariableop_32_feature_maps_feature_map_norm4_moving_variance:	А#
assignvariableop_33_total: #
assignvariableop_34_count: %
assignvariableop_35_total_1: %
assignvariableop_36_count_1: V
<assignvariableop_37_digit_caps_digit_caps_transform_tensor_m:
L
6assignvariableop_38_digit_caps_digit_caps_log_priors_m:
U
;assignvariableop_39_feature_maps_feature_map_conv1_kernel_m: G
9assignvariableop_40_feature_maps_feature_map_conv1_bias_m: H
:assignvariableop_41_feature_maps_feature_map_norm1_gamma_m: G
9assignvariableop_42_feature_maps_feature_map_norm1_beta_m: U
;assignvariableop_43_feature_maps_feature_map_conv2_kernel_m: @G
9assignvariableop_44_feature_maps_feature_map_conv2_bias_m:@H
:assignvariableop_45_feature_maps_feature_map_norm2_gamma_m:@G
9assignvariableop_46_feature_maps_feature_map_norm2_beta_m:@U
;assignvariableop_47_feature_maps_feature_map_conv3_kernel_m:@@G
9assignvariableop_48_feature_maps_feature_map_conv3_bias_m:@H
:assignvariableop_49_feature_maps_feature_map_norm3_gamma_m:@G
9assignvariableop_50_feature_maps_feature_map_norm3_beta_m:@V
;assignvariableop_51_feature_maps_feature_map_conv4_kernel_m:@АH
9assignvariableop_52_feature_maps_feature_map_conv4_bias_m:	АI
:assignvariableop_53_feature_maps_feature_map_norm4_gamma_m:	АH
9assignvariableop_54_feature_maps_feature_map_norm4_beta_m:	АV
;assignvariableop_55_primary_caps_primary_cap_dconv_kernel_m:		АH
9assignvariableop_56_primary_caps_primary_cap_dconv_bias_m:	АV
<assignvariableop_57_digit_caps_digit_caps_transform_tensor_v:
L
6assignvariableop_58_digit_caps_digit_caps_log_priors_v:
U
;assignvariableop_59_feature_maps_feature_map_conv1_kernel_v: G
9assignvariableop_60_feature_maps_feature_map_conv1_bias_v: H
:assignvariableop_61_feature_maps_feature_map_norm1_gamma_v: G
9assignvariableop_62_feature_maps_feature_map_norm1_beta_v: U
;assignvariableop_63_feature_maps_feature_map_conv2_kernel_v: @G
9assignvariableop_64_feature_maps_feature_map_conv2_bias_v:@H
:assignvariableop_65_feature_maps_feature_map_norm2_gamma_v:@G
9assignvariableop_66_feature_maps_feature_map_norm2_beta_v:@U
;assignvariableop_67_feature_maps_feature_map_conv3_kernel_v:@@G
9assignvariableop_68_feature_maps_feature_map_conv3_bias_v:@H
:assignvariableop_69_feature_maps_feature_map_norm3_gamma_v:@G
9assignvariableop_70_feature_maps_feature_map_norm3_beta_v:@V
;assignvariableop_71_feature_maps_feature_map_conv4_kernel_v:@АH
9assignvariableop_72_feature_maps_feature_map_conv4_bias_v:	АI
:assignvariableop_73_feature_maps_feature_map_norm4_gamma_v:	АH
9assignvariableop_74_feature_maps_feature_map_norm4_beta_v:	АV
;assignvariableop_75_primary_caps_primary_cap_dconv_kernel_v:		АH
9assignvariableop_76_primary_caps_primary_cap_dconv_bias_v:	А
identity_78ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_52вAssignVariableOp_53вAssignVariableOp_54вAssignVariableOp_55вAssignVariableOp_56вAssignVariableOp_57вAssignVariableOp_58вAssignVariableOp_59вAssignVariableOp_6вAssignVariableOp_60вAssignVariableOp_61вAssignVariableOp_62вAssignVariableOp_63вAssignVariableOp_64вAssignVariableOp_65вAssignVariableOp_66вAssignVariableOp_67вAssignVariableOp_68вAssignVariableOp_69вAssignVariableOp_7вAssignVariableOp_70вAssignVariableOp_71вAssignVariableOp_72вAssignVariableOp_73вAssignVariableOp_74вAssignVariableOp_75вAssignVariableOp_76вAssignVariableOp_8вAssignVariableOp_9ь(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*°'
valueю'Bы'NBKlayer_with_weights-2/digit_caps_transform_tensor/.ATTRIBUTES/VARIABLE_VALUEBElayer_with_weights-2/digit_caps_log_priors/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBglayer_with_weights-2/digit_caps_transform_tensor/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBalayer_with_weights-2/digit_caps_log_priors/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesн
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueзBдNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices┤
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
╕::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity╢
AssignVariableOpAssignVariableOp7assignvariableop_digit_caps_digit_caps_transform_tensorIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1╕
AssignVariableOp_1AssignVariableOp3assignvariableop_1_digit_caps_digit_caps_log_priorsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╜
AssignVariableOp_7AssignVariableOp8assignvariableop_7_feature_maps_feature_map_conv1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╗
AssignVariableOp_8AssignVariableOp6assignvariableop_8_feature_maps_feature_map_conv1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╝
AssignVariableOp_9AssignVariableOp7assignvariableop_9_feature_maps_feature_map_norm1_gammaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┐
AssignVariableOp_10AssignVariableOp7assignvariableop_10_feature_maps_feature_map_norm1_betaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┴
AssignVariableOp_11AssignVariableOp9assignvariableop_11_feature_maps_feature_map_conv2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┐
AssignVariableOp_12AssignVariableOp7assignvariableop_12_feature_maps_feature_map_conv2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13└
AssignVariableOp_13AssignVariableOp8assignvariableop_13_feature_maps_feature_map_norm2_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14┐
AssignVariableOp_14AssignVariableOp7assignvariableop_14_feature_maps_feature_map_norm2_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15┴
AssignVariableOp_15AssignVariableOp9assignvariableop_15_feature_maps_feature_map_conv3_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16┐
AssignVariableOp_16AssignVariableOp7assignvariableop_16_feature_maps_feature_map_conv3_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17└
AssignVariableOp_17AssignVariableOp8assignvariableop_17_feature_maps_feature_map_norm3_gammaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18┐
AssignVariableOp_18AssignVariableOp7assignvariableop_18_feature_maps_feature_map_norm3_betaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19┴
AssignVariableOp_19AssignVariableOp9assignvariableop_19_feature_maps_feature_map_conv4_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┐
AssignVariableOp_20AssignVariableOp7assignvariableop_20_feature_maps_feature_map_conv4_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21└
AssignVariableOp_21AssignVariableOp8assignvariableop_21_feature_maps_feature_map_norm4_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22┐
AssignVariableOp_22AssignVariableOp7assignvariableop_22_feature_maps_feature_map_norm4_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23┴
AssignVariableOp_23AssignVariableOp9assignvariableop_23_primary_caps_primary_cap_dconv_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24┐
AssignVariableOp_24AssignVariableOp7assignvariableop_24_primary_caps_primary_cap_dconv_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25╞
AssignVariableOp_25AssignVariableOp>assignvariableop_25_feature_maps_feature_map_norm1_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26╩
AssignVariableOp_26AssignVariableOpBassignvariableop_26_feature_maps_feature_map_norm1_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╞
AssignVariableOp_27AssignVariableOp>assignvariableop_27_feature_maps_feature_map_norm2_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╩
AssignVariableOp_28AssignVariableOpBassignvariableop_28_feature_maps_feature_map_norm2_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╞
AssignVariableOp_29AssignVariableOp>assignvariableop_29_feature_maps_feature_map_norm3_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╩
AssignVariableOp_30AssignVariableOpBassignvariableop_30_feature_maps_feature_map_norm3_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╞
AssignVariableOp_31AssignVariableOp>assignvariableop_31_feature_maps_feature_map_norm4_moving_meanIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╩
AssignVariableOp_32AssignVariableOpBassignvariableop_32_feature_maps_feature_map_norm4_moving_varianceIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33б
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34б
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35г
AssignVariableOp_35AssignVariableOpassignvariableop_35_total_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36г
AssignVariableOp_36AssignVariableOpassignvariableop_36_count_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37─
AssignVariableOp_37AssignVariableOp<assignvariableop_37_digit_caps_digit_caps_transform_tensor_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38╛
AssignVariableOp_38AssignVariableOp6assignvariableop_38_digit_caps_digit_caps_log_priors_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39├
AssignVariableOp_39AssignVariableOp;assignvariableop_39_feature_maps_feature_map_conv1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40┴
AssignVariableOp_40AssignVariableOp9assignvariableop_40_feature_maps_feature_map_conv1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41┬
AssignVariableOp_41AssignVariableOp:assignvariableop_41_feature_maps_feature_map_norm1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42┴
AssignVariableOp_42AssignVariableOp9assignvariableop_42_feature_maps_feature_map_norm1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43├
AssignVariableOp_43AssignVariableOp;assignvariableop_43_feature_maps_feature_map_conv2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44┴
AssignVariableOp_44AssignVariableOp9assignvariableop_44_feature_maps_feature_map_conv2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45┬
AssignVariableOp_45AssignVariableOp:assignvariableop_45_feature_maps_feature_map_norm2_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46┴
AssignVariableOp_46AssignVariableOp9assignvariableop_46_feature_maps_feature_map_norm2_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47├
AssignVariableOp_47AssignVariableOp;assignvariableop_47_feature_maps_feature_map_conv3_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48┴
AssignVariableOp_48AssignVariableOp9assignvariableop_48_feature_maps_feature_map_conv3_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49┬
AssignVariableOp_49AssignVariableOp:assignvariableop_49_feature_maps_feature_map_norm3_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50┴
AssignVariableOp_50AssignVariableOp9assignvariableop_50_feature_maps_feature_map_norm3_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51├
AssignVariableOp_51AssignVariableOp;assignvariableop_51_feature_maps_feature_map_conv4_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52┴
AssignVariableOp_52AssignVariableOp9assignvariableop_52_feature_maps_feature_map_conv4_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53┬
AssignVariableOp_53AssignVariableOp:assignvariableop_53_feature_maps_feature_map_norm4_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54┴
AssignVariableOp_54AssignVariableOp9assignvariableop_54_feature_maps_feature_map_norm4_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55├
AssignVariableOp_55AssignVariableOp;assignvariableop_55_primary_caps_primary_cap_dconv_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56┴
AssignVariableOp_56AssignVariableOp9assignvariableop_56_primary_caps_primary_cap_dconv_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57─
AssignVariableOp_57AssignVariableOp<assignvariableop_57_digit_caps_digit_caps_transform_tensor_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58╛
AssignVariableOp_58AssignVariableOp6assignvariableop_58_digit_caps_digit_caps_log_priors_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59├
AssignVariableOp_59AssignVariableOp;assignvariableop_59_feature_maps_feature_map_conv1_kernel_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60┴
AssignVariableOp_60AssignVariableOp9assignvariableop_60_feature_maps_feature_map_conv1_bias_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61┬
AssignVariableOp_61AssignVariableOp:assignvariableop_61_feature_maps_feature_map_norm1_gamma_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62┴
AssignVariableOp_62AssignVariableOp9assignvariableop_62_feature_maps_feature_map_norm1_beta_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63├
AssignVariableOp_63AssignVariableOp;assignvariableop_63_feature_maps_feature_map_conv2_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64┴
AssignVariableOp_64AssignVariableOp9assignvariableop_64_feature_maps_feature_map_conv2_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65┬
AssignVariableOp_65AssignVariableOp:assignvariableop_65_feature_maps_feature_map_norm2_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66┴
AssignVariableOp_66AssignVariableOp9assignvariableop_66_feature_maps_feature_map_norm2_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67├
AssignVariableOp_67AssignVariableOp;assignvariableop_67_feature_maps_feature_map_conv3_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68┴
AssignVariableOp_68AssignVariableOp9assignvariableop_68_feature_maps_feature_map_conv3_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69┬
AssignVariableOp_69AssignVariableOp:assignvariableop_69_feature_maps_feature_map_norm3_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70┴
AssignVariableOp_70AssignVariableOp9assignvariableop_70_feature_maps_feature_map_norm3_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71├
AssignVariableOp_71AssignVariableOp;assignvariableop_71_feature_maps_feature_map_conv4_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72┴
AssignVariableOp_72AssignVariableOp9assignvariableop_72_feature_maps_feature_map_conv4_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73┬
AssignVariableOp_73AssignVariableOp:assignvariableop_73_feature_maps_feature_map_norm4_gamma_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74┴
AssignVariableOp_74AssignVariableOp9assignvariableop_74_feature_maps_feature_map_norm4_beta_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75├
AssignVariableOp_75AssignVariableOp;assignvariableop_75_primary_caps_primary_cap_dconv_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76┴
AssignVariableOp_76AssignVariableOp9assignvariableop_76_primary_caps_primary_cap_dconv_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_769
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp№
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_77f
Identity_78IdentityIdentity_77:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_78ф
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_78Identity_78:output:0*▒
_input_shapesЯ
Ь: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
■y
щ
F__inference_feature_maps_layer_call_and_return_conditional_losses_7341
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@А@
1feature_map_conv4_biasadd_readvariableop_resource:	А8
)feature_map_norm4_readvariableop_resource:	А:
+feature_map_norm4_readvariableop_1_resource:	АI
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АK
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв(feature_map_conv1/BiasAdd/ReadVariableOpв'feature_map_conv1/Conv2D/ReadVariableOpв(feature_map_conv2/BiasAdd/ReadVariableOpв'feature_map_conv2/Conv2D/ReadVariableOpв(feature_map_conv3/BiasAdd/ReadVariableOpв'feature_map_conv3/Conv2D/ReadVariableOpв(feature_map_conv4/BiasAdd/ReadVariableOpв'feature_map_conv4/Conv2D/ReadVariableOpв1feature_map_norm1/FusedBatchNormV3/ReadVariableOpв3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm1/ReadVariableOpв"feature_map_norm1/ReadVariableOp_1в1feature_map_norm2/FusedBatchNormV3/ReadVariableOpв3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm2/ReadVariableOpв"feature_map_norm2/ReadVariableOp_1в1feature_map_norm3/FusedBatchNormV3/ReadVariableOpв3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm3/ReadVariableOpв"feature_map_norm3/ReadVariableOp_1в1feature_map_norm4/FusedBatchNormV3/ReadVariableOpв3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm4/ReadVariableOpв"feature_map_norm4/ReadVariableOp_1╦
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'feature_map_conv1/Conv2D/ReadVariableOpр
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
feature_map_conv1/Conv2D┬
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(feature_map_conv1/BiasAdd/ReadVariableOp╨
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
feature_map_conv1/BiasAddЦ
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
feature_map_conv1/Reluк
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02"
 feature_map_norm1/ReadVariableOp░
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"feature_map_norm1/ReadVariableOp_1▌
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
is_training( 2$
"feature_map_norm1/FusedBatchNormV3╦
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'feature_map_conv2/Conv2D/ReadVariableOp·
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv2/Conv2D┬
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv2/BiasAdd/ReadVariableOp╨
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/BiasAddЦ
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/Reluк
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm2/ReadVariableOp░
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm2/ReadVariableOp_1▌
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm2/FusedBatchNormV3╦
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'feature_map_conv3/Conv2D/ReadVariableOp·
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv3/Conv2D┬
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv3/BiasAdd/ReadVariableOp╨
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/BiasAddЦ
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/Reluк
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm3/ReadVariableOp░
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm3/ReadVariableOp_1▌
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1╘
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm3/FusedBatchNormV3╠
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02)
'feature_map_conv4/Conv2D/ReadVariableOp√
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2
feature_map_conv4/Conv2D├
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(feature_map_conv4/BiasAdd/ReadVariableOp╤
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/BiasAddЧ
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/Reluл
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 feature_map_norm4/ReadVariableOp▒
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02$
"feature_map_norm4/ReadVariableOp_1▐
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype023
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpф
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype025
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1┘
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
is_training( 2$
"feature_map_norm4/FusedBatchNormV3К
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         		А2

Identityъ
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp2^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
▌
Ц
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_6764

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИвFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1з
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpн
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1▄
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oГ:*
is_training( 2
FusedBatchNormV3Й
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            2

Identity╕
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ь[
∙
D__inference_digit_caps_layer_call_and_return_conditional_losses_7557
primary_caps+
map_while_input_6:
	
mul_x3
add_3_readvariableop_resource:

identityИвadd_3/ReadVariableOpв	map/whileb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dimЗ

ExpandDims
ExpandDimsprimary_capsExpandDims/dim:output:0*
T0*/
_output_shapes
:         2

ExpandDimsy
Tile/multiplesConst*
_output_shapes
:*
dtype0*%
valueB"   
         2
Tile/multiples|
TileTileExpandDims:output:0Tile/multiples:output:0*
T0*/
_output_shapes
:         
2
Tilew
digit_cap_inputs/dimConst*
_output_shapes
: *
dtype0*
valueB :
         2
digit_cap_inputs/dimЮ
digit_cap_inputs
ExpandDimsTile:output:0digit_cap_inputs/dim:output:0*
T0*3
_output_shapes!
:         
2
digit_cap_inputs_
	map/ShapeShapedigit_cap_inputs:output:0*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stackА
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1А
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2·
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_sliceН
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2!
map/TensorArrayV2/element_shape└
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2╧
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeР
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensordigit_cap_inputs:output:0Bmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/ConstС
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2#
!map/TensorArrayV2_1/element_shape╞
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counterХ
	map/whileWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0map_while_input_6*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*"
_output_shapes
: : : : : : : *#
_read_only_resource_inputs
*
_stateful_parallelism( *
bodyR
map_while_body_7452*
condR
map_while_cond_7451*!
output_shapes
: : : : : : : 2
	map/while┼
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*%
valueB"
            26
4map/TensorArrayV2Stack/TensorListStack/element_shapeА
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*3
_output_shapes!
:         
*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStack─
digit_cap_predictionsSqueeze/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*/
_output_shapes
:         
*
squeeze_dims

         2
digit_cap_predictions─
digit_cap_attentionsBatchMatMulV2digit_cap_predictions:output:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
*
adj_y(2
digit_cap_attentionsq
mulMulmul_xdigit_cap_attentions:output:0*
T0*/
_output_shapes
:         
2
muly
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
■        2
Sum/reduction_indicesЕ
SumSummul:z:0Sum/reduction_indices:output:0*
T0*/
_output_shapes
:         
*
	keep_dims(2
SumN
RankConst*
_output_shapes
: *
dtype0*
value	B :2
RankY
add/xConst*
_output_shapes
: *
dtype0*
valueB :
■        2
add/xS
addAddV2add/x:output:0Rank:output:0*
T0*
_output_shapes
: 2
addR
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
Rank_1P
mod/yConst*
_output_shapes
: *
dtype0*
value	B :2
mod/yP
modFloorModadd:z:0mod/y:output:0*
T0*
_output_shapes
: 2
modP
Sub/yConst*
_output_shapes
: *
dtype0*
value	B :2
Sub/yS
SubSubRank_1:output:0Sub/y:output:0*
T0*
_output_shapes
: 2
Sub\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltah
rangeRangerange/start:output:0mod:z:0range/delta:output:0*
_output_shapes
:2
rangeT
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yS
add_1AddV2mod:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/deltaa
range_1Range	add_1:z:0Sub:z:0range_1/delta:output:0*
_output_shapes
: 2	
range_1a
concat/values_1PackSub:z:0*
N*
T0*
_output_shapes
:2
concat/values_1a
concat/values_3Packmod:z:0*
N*
T0*
_output_shapes
:2
concat/values_3\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat/axis╢
concatConcatV2range:output:0concat/values_1:output:0range_1:output:0concat/values_3:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:2
concat|
	transpose	TransposeSum:output:0concat:output:0*
T0*/
_output_shapes
:         
2
	transposef
SoftmaxSoftmaxtranspose:y:0*
T0*/
_output_shapes
:         
2	
SoftmaxT
Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
Sub_1/yY
Sub_1SubRank_1:output:0Sub_1/y:output:0*
T0*
_output_shapes
: 2
Sub_1`
range_2/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_2/start`
range_2/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_2/deltap
range_2Rangerange_2/start:output:0mod:z:0range_2/delta:output:0*
_output_shapes
:2	
range_2T
add_2/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_2/yS
add_2AddV2mod:z:0add_2/y:output:0*
T0*
_output_shapes
: 2
add_2`
range_3/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_3/deltac
range_3Range	add_2:z:0	Sub_1:z:0range_3/delta:output:0*
_output_shapes
: 2	
range_3g
concat_1/values_1Pack	Sub_1:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_1e
concat_1/values_3Packmod:z:0*
N*
T0*
_output_shapes
:2
concat_1/values_3`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis┬
concat_1ConcatV2range_2:output:0concat_1/values_1:output:0range_3:output:0concat_1/values_3:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1п
digit_cap_coupling_coefficients	TransposeSoftmax:softmax:0concat_1:output:0*
T0*/
_output_shapes
:         
2!
digit_cap_coupling_coefficientsО
add_3/ReadVariableOpReadVariableOpadd_3_readvariableop_resource*"
_output_shapes
:
*
dtype02
add_3/ReadVariableOpФ
add_3AddV2#digit_cap_coupling_coefficients:y:0add_3/ReadVariableOp:value:0*
T0*/
_output_shapes
:         
2
add_3Ж
MatMulBatchMatMulV2	add_3:z:0digit_cap_predictions:output:0*
T0*/
_output_shapes
:         
2
MatMulД
SqueezeSqueezeMatMul:output:0*
T0*+
_output_shapes
:         
*
squeeze_dims

■        2	
SqueezeЧ
digit_cap_squash/norm/mulMulSqueeze:output:0Squeeze:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/norm/mulн
+digit_cap_squash/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+digit_cap_squash/norm/Sum/reduction_indices┘
digit_cap_squash/norm/SumSumdigit_cap_squash/norm/mul:z:04digit_cap_squash/norm/Sum/reduction_indices:output:0*
T0*+
_output_shapes
:         
*
	keep_dims(2
digit_cap_squash/norm/SumЪ
digit_cap_squash/norm/SqrtSqrt"digit_cap_squash/norm/Sum:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/norm/SqrtЙ
digit_cap_squash/ExpExpdigit_cap_squash/norm/Sqrt:y:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/Exp}
digit_cap_squash/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
digit_cap_squash/truediv/x┤
digit_cap_squash/truedivRealDiv#digit_cap_squash/truediv/x:output:0digit_cap_squash/Exp:y:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/truedivu
digit_cap_squash/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
digit_cap_squash/sub/xи
digit_cap_squash/subSubdigit_cap_squash/sub/x:output:0digit_cap_squash/truediv:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/subu
digit_cap_squash/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓32
digit_cap_squash/add/yм
digit_cap_squash/addAddV2digit_cap_squash/norm/Sqrt:y:0digit_cap_squash/add/y:output:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/addе
digit_cap_squash/truediv_1RealDivSqueeze:output:0digit_cap_squash/add:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/truediv_1г
digit_cap_squash/mulMuldigit_cap_squash/sub:z:0digit_cap_squash/truediv_1:z:0*
T0*+
_output_shapes
:         
2
digit_cap_squash/mulw
IdentityIdentitydigit_cap_squash/mul:z:0^NoOp*
T0*+
_output_shapes
:         
2

Identityq
NoOpNoOp^add_3/ReadVariableOp
^map/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2,
add_3/ReadVariableOpadd_3/ReadVariableOp2
	map/while	map/while:Y U
+
_output_shapes
:         
&
_user_specified_nameprimary_caps:

_output_shapes
: 
▐Х
Й
F__inference_feature_maps_layer_call_and_return_conditional_losses_7824
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@А@
1feature_map_conv4_biasadd_readvariableop_resource:	А8
)feature_map_norm4_readvariableop_resource:	А:
+feature_map_norm4_readvariableop_1_resource:	АI
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АK
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв(feature_map_conv1/BiasAdd/ReadVariableOpв'feature_map_conv1/Conv2D/ReadVariableOpв(feature_map_conv2/BiasAdd/ReadVariableOpв'feature_map_conv2/Conv2D/ReadVariableOpв(feature_map_conv3/BiasAdd/ReadVariableOpв'feature_map_conv3/Conv2D/ReadVariableOpв(feature_map_conv4/BiasAdd/ReadVariableOpв'feature_map_conv4/Conv2D/ReadVariableOpв feature_map_norm1/AssignNewValueв"feature_map_norm1/AssignNewValue_1в1feature_map_norm1/FusedBatchNormV3/ReadVariableOpв3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm1/ReadVariableOpв"feature_map_norm1/ReadVariableOp_1в feature_map_norm2/AssignNewValueв"feature_map_norm2/AssignNewValue_1в1feature_map_norm2/FusedBatchNormV3/ReadVariableOpв3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm2/ReadVariableOpв"feature_map_norm2/ReadVariableOp_1в feature_map_norm3/AssignNewValueв"feature_map_norm3/AssignNewValue_1в1feature_map_norm3/FusedBatchNormV3/ReadVariableOpв3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm3/ReadVariableOpв"feature_map_norm3/ReadVariableOp_1в feature_map_norm4/AssignNewValueв"feature_map_norm4/AssignNewValue_1в1feature_map_norm4/FusedBatchNormV3/ReadVariableOpв3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm4/ReadVariableOpв"feature_map_norm4/ReadVariableOp_1╦
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'feature_map_conv1/Conv2D/ReadVariableOpр
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
feature_map_conv1/Conv2D┬
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(feature_map_conv1/BiasAdd/ReadVariableOp╨
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
feature_map_conv1/BiasAddЦ
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
feature_map_conv1/Reluк
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02"
 feature_map_norm1/ReadVariableOp░
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"feature_map_norm1/ReadVariableOp_1▌
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm1/FusedBatchNormV3Ь
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm1/AssignNewValueи
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm1/AssignNewValue_1╦
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'feature_map_conv2/Conv2D/ReadVariableOp·
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv2/Conv2D┬
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv2/BiasAdd/ReadVariableOp╨
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/BiasAddЦ
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/Reluк
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm2/ReadVariableOp░
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm2/ReadVariableOp_1▌
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm2/FusedBatchNormV3Ь
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm2/AssignNewValueи
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm2/AssignNewValue_1╦
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'feature_map_conv3/Conv2D/ReadVariableOp·
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv3/Conv2D┬
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv3/BiasAdd/ReadVariableOp╨
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/BiasAddЦ
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/Reluк
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm3/ReadVariableOp░
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm3/ReadVariableOp_1▌
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm3/FusedBatchNormV3Ь
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm3/AssignNewValueи
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm3/AssignNewValue_1╠
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02)
'feature_map_conv4/Conv2D/ReadVariableOp√
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2
feature_map_conv4/Conv2D├
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(feature_map_conv4/BiasAdd/ReadVariableOp╤
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/BiasAddЧ
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/Reluл
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 feature_map_norm4/ReadVariableOp▒
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02$
"feature_map_norm4/ReadVariableOp_1▐
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype023
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpф
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype025
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ч
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm4/FusedBatchNormV3Ь
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm4/AssignNewValueи
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm4/AssignNewValue_1К
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         		А2

IdentityК
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2D
 feature_map_norm1/AssignNewValue feature_map_norm1/AssignNewValue2H
"feature_map_norm1/AssignNewValue_1"feature_map_norm1/AssignNewValue_12f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12D
 feature_map_norm2/AssignNewValue feature_map_norm2/AssignNewValue2H
"feature_map_norm2/AssignNewValue_1"feature_map_norm2/AssignNewValue_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12D
 feature_map_norm3/AssignNewValue feature_map_norm3/AssignNewValue2H
"feature_map_norm3/AssignNewValue_1"feature_map_norm3/AssignNewValue_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12D
 feature_map_norm4/AssignNewValue feature_map_norm4/AssignNewValue2H
"feature_map_norm4/AssignNewValue_1"feature_map_norm4/AssignNewValue_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
▐Х
Й
F__inference_feature_maps_layer_call_and_return_conditional_losses_9231
input_imagesJ
0feature_map_conv1_conv2d_readvariableop_resource: ?
1feature_map_conv1_biasadd_readvariableop_resource: 7
)feature_map_norm1_readvariableop_resource: 9
+feature_map_norm1_readvariableop_1_resource: H
:feature_map_norm1_fusedbatchnormv3_readvariableop_resource: J
<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource: J
0feature_map_conv2_conv2d_readvariableop_resource: @?
1feature_map_conv2_biasadd_readvariableop_resource:@7
)feature_map_norm2_readvariableop_resource:@9
+feature_map_norm2_readvariableop_1_resource:@H
:feature_map_norm2_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource:@J
0feature_map_conv3_conv2d_readvariableop_resource:@@?
1feature_map_conv3_biasadd_readvariableop_resource:@7
)feature_map_norm3_readvariableop_resource:@9
+feature_map_norm3_readvariableop_1_resource:@H
:feature_map_norm3_fusedbatchnormv3_readvariableop_resource:@J
<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource:@K
0feature_map_conv4_conv2d_readvariableop_resource:@А@
1feature_map_conv4_biasadd_readvariableop_resource:	А8
)feature_map_norm4_readvariableop_resource:	А:
+feature_map_norm4_readvariableop_1_resource:	АI
:feature_map_norm4_fusedbatchnormv3_readvariableop_resource:	АK
<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource:	А
identityИв(feature_map_conv1/BiasAdd/ReadVariableOpв'feature_map_conv1/Conv2D/ReadVariableOpв(feature_map_conv2/BiasAdd/ReadVariableOpв'feature_map_conv2/Conv2D/ReadVariableOpв(feature_map_conv3/BiasAdd/ReadVariableOpв'feature_map_conv3/Conv2D/ReadVariableOpв(feature_map_conv4/BiasAdd/ReadVariableOpв'feature_map_conv4/Conv2D/ReadVariableOpв feature_map_norm1/AssignNewValueв"feature_map_norm1/AssignNewValue_1в1feature_map_norm1/FusedBatchNormV3/ReadVariableOpв3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm1/ReadVariableOpв"feature_map_norm1/ReadVariableOp_1в feature_map_norm2/AssignNewValueв"feature_map_norm2/AssignNewValue_1в1feature_map_norm2/FusedBatchNormV3/ReadVariableOpв3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm2/ReadVariableOpв"feature_map_norm2/ReadVariableOp_1в feature_map_norm3/AssignNewValueв"feature_map_norm3/AssignNewValue_1в1feature_map_norm3/FusedBatchNormV3/ReadVariableOpв3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm3/ReadVariableOpв"feature_map_norm3/ReadVariableOp_1в feature_map_norm4/AssignNewValueв"feature_map_norm4/AssignNewValue_1в1feature_map_norm4/FusedBatchNormV3/ReadVariableOpв3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1в feature_map_norm4/ReadVariableOpв"feature_map_norm4/ReadVariableOp_1╦
'feature_map_conv1/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'feature_map_conv1/Conv2D/ReadVariableOpр
feature_map_conv1/Conv2DConv2Dinput_images/feature_map_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:          *
paddingVALID*
strides
2
feature_map_conv1/Conv2D┬
(feature_map_conv1/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(feature_map_conv1/BiasAdd/ReadVariableOp╨
feature_map_conv1/BiasAddBiasAdd!feature_map_conv1/Conv2D:output:00feature_map_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:          2
feature_map_conv1/BiasAddЦ
feature_map_conv1/ReluRelu"feature_map_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:          2
feature_map_conv1/Reluк
 feature_map_norm1/ReadVariableOpReadVariableOp)feature_map_norm1_readvariableop_resource*
_output_shapes
: *
dtype02"
 feature_map_norm1/ReadVariableOp░
"feature_map_norm1/ReadVariableOp_1ReadVariableOp+feature_map_norm1_readvariableop_1_resource*
_output_shapes
: *
dtype02$
"feature_map_norm1/ReadVariableOp_1▌
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype023
1feature_map_norm1/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype025
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm1/FusedBatchNormV3FusedBatchNormV3$feature_map_conv1/Relu:activations:0(feature_map_norm1/ReadVariableOp:value:0*feature_map_norm1/ReadVariableOp_1:value:09feature_map_norm1/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm1/FusedBatchNormV3Ь
 feature_map_norm1/AssignNewValueAssignVariableOp:feature_map_norm1_fusedbatchnormv3_readvariableop_resource/feature_map_norm1/FusedBatchNormV3:batch_mean:02^feature_map_norm1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm1/AssignNewValueи
"feature_map_norm1/AssignNewValue_1AssignVariableOp<feature_map_norm1_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm1/FusedBatchNormV3:batch_variance:04^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm1/AssignNewValue_1╦
'feature_map_conv2/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'feature_map_conv2/Conv2D/ReadVariableOp·
feature_map_conv2/Conv2DConv2D&feature_map_norm1/FusedBatchNormV3:y:0/feature_map_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv2/Conv2D┬
(feature_map_conv2/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv2/BiasAdd/ReadVariableOp╨
feature_map_conv2/BiasAddBiasAdd!feature_map_conv2/Conv2D:output:00feature_map_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/BiasAddЦ
feature_map_conv2/ReluRelu"feature_map_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv2/Reluк
 feature_map_norm2/ReadVariableOpReadVariableOp)feature_map_norm2_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm2/ReadVariableOp░
"feature_map_norm2/ReadVariableOp_1ReadVariableOp+feature_map_norm2_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm2/ReadVariableOp_1▌
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm2/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm2/FusedBatchNormV3FusedBatchNormV3$feature_map_conv2/Relu:activations:0(feature_map_norm2/ReadVariableOp:value:0*feature_map_norm2/ReadVariableOp_1:value:09feature_map_norm2/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm2/FusedBatchNormV3Ь
 feature_map_norm2/AssignNewValueAssignVariableOp:feature_map_norm2_fusedbatchnormv3_readvariableop_resource/feature_map_norm2/FusedBatchNormV3:batch_mean:02^feature_map_norm2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm2/AssignNewValueи
"feature_map_norm2/AssignNewValue_1AssignVariableOp<feature_map_norm2_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm2/FusedBatchNormV3:batch_variance:04^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm2/AssignNewValue_1╦
'feature_map_conv3/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'feature_map_conv3/Conv2D/ReadVariableOp·
feature_map_conv3/Conv2DConv2D&feature_map_norm2/FusedBatchNormV3:y:0/feature_map_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
2
feature_map_conv3/Conv2D┬
(feature_map_conv3/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(feature_map_conv3/BiasAdd/ReadVariableOp╨
feature_map_conv3/BiasAddBiasAdd!feature_map_conv3/Conv2D:output:00feature_map_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/BiasAddЦ
feature_map_conv3/ReluRelu"feature_map_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:         @2
feature_map_conv3/Reluк
 feature_map_norm3/ReadVariableOpReadVariableOp)feature_map_norm3_readvariableop_resource*
_output_shapes
:@*
dtype02"
 feature_map_norm3/ReadVariableOp░
"feature_map_norm3/ReadVariableOp_1ReadVariableOp+feature_map_norm3_readvariableop_1_resource*
_output_shapes
:@*
dtype02$
"feature_map_norm3/ReadVariableOp_1▌
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype023
1feature_map_norm3/FusedBatchNormV3/ReadVariableOpу
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1т
"feature_map_norm3/FusedBatchNormV3FusedBatchNormV3$feature_map_conv3/Relu:activations:0(feature_map_norm3/ReadVariableOp:value:0*feature_map_norm3/ReadVariableOp_1:value:09feature_map_norm3/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm3/FusedBatchNormV3Ь
 feature_map_norm3/AssignNewValueAssignVariableOp:feature_map_norm3_fusedbatchnormv3_readvariableop_resource/feature_map_norm3/FusedBatchNormV3:batch_mean:02^feature_map_norm3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm3/AssignNewValueи
"feature_map_norm3/AssignNewValue_1AssignVariableOp<feature_map_norm3_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm3/FusedBatchNormV3:batch_variance:04^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm3/AssignNewValue_1╠
'feature_map_conv4/Conv2D/ReadVariableOpReadVariableOp0feature_map_conv4_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype02)
'feature_map_conv4/Conv2D/ReadVariableOp√
feature_map_conv4/Conv2DConv2D&feature_map_norm3/FusedBatchNormV3:y:0/feature_map_conv4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А*
paddingVALID*
strides
2
feature_map_conv4/Conv2D├
(feature_map_conv4/BiasAdd/ReadVariableOpReadVariableOp1feature_map_conv4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02*
(feature_map_conv4/BiasAdd/ReadVariableOp╤
feature_map_conv4/BiasAddBiasAdd!feature_map_conv4/Conv2D:output:00feature_map_conv4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/BiasAddЧ
feature_map_conv4/ReluRelu"feature_map_conv4/BiasAdd:output:0*
T0*0
_output_shapes
:         		А2
feature_map_conv4/Reluл
 feature_map_norm4/ReadVariableOpReadVariableOp)feature_map_norm4_readvariableop_resource*
_output_shapes	
:А*
dtype02"
 feature_map_norm4/ReadVariableOp▒
"feature_map_norm4/ReadVariableOp_1ReadVariableOp+feature_map_norm4_readvariableop_1_resource*
_output_shapes	
:А*
dtype02$
"feature_map_norm4/ReadVariableOp_1▐
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpReadVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype023
1feature_map_norm4/FusedBatchNormV3/ReadVariableOpф
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype025
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1ч
"feature_map_norm4/FusedBatchNormV3FusedBatchNormV3$feature_map_conv4/Relu:activations:0(feature_map_norm4/ReadVariableOp:value:0*feature_map_norm4/ReadVariableOp_1:value:09feature_map_norm4/FusedBatchNormV3/ReadVariableOp:value:0;feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:         		А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2$
"feature_map_norm4/FusedBatchNormV3Ь
 feature_map_norm4/AssignNewValueAssignVariableOp:feature_map_norm4_fusedbatchnormv3_readvariableop_resource/feature_map_norm4/FusedBatchNormV3:batch_mean:02^feature_map_norm4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02"
 feature_map_norm4/AssignNewValueи
"feature_map_norm4/AssignNewValue_1AssignVariableOp<feature_map_norm4_fusedbatchnormv3_readvariableop_1_resource3feature_map_norm4/FusedBatchNormV3:batch_variance:04^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02$
"feature_map_norm4/AssignNewValue_1К
IdentityIdentity&feature_map_norm4/FusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:         		А2

IdentityК
NoOpNoOp)^feature_map_conv1/BiasAdd/ReadVariableOp(^feature_map_conv1/Conv2D/ReadVariableOp)^feature_map_conv2/BiasAdd/ReadVariableOp(^feature_map_conv2/Conv2D/ReadVariableOp)^feature_map_conv3/BiasAdd/ReadVariableOp(^feature_map_conv3/Conv2D/ReadVariableOp)^feature_map_conv4/BiasAdd/ReadVariableOp(^feature_map_conv4/Conv2D/ReadVariableOp!^feature_map_norm1/AssignNewValue#^feature_map_norm1/AssignNewValue_12^feature_map_norm1/FusedBatchNormV3/ReadVariableOp4^feature_map_norm1/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm1/ReadVariableOp#^feature_map_norm1/ReadVariableOp_1!^feature_map_norm2/AssignNewValue#^feature_map_norm2/AssignNewValue_12^feature_map_norm2/FusedBatchNormV3/ReadVariableOp4^feature_map_norm2/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm2/ReadVariableOp#^feature_map_norm2/ReadVariableOp_1!^feature_map_norm3/AssignNewValue#^feature_map_norm3/AssignNewValue_12^feature_map_norm3/FusedBatchNormV3/ReadVariableOp4^feature_map_norm3/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm3/ReadVariableOp#^feature_map_norm3/ReadVariableOp_1!^feature_map_norm4/AssignNewValue#^feature_map_norm4/AssignNewValue_12^feature_map_norm4/FusedBatchNormV3/ReadVariableOp4^feature_map_norm4/FusedBatchNormV3/ReadVariableOp_1!^feature_map_norm4/ReadVariableOp#^feature_map_norm4/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:         : : : : : : : : : : : : : : : : : : : : : : : : 2T
(feature_map_conv1/BiasAdd/ReadVariableOp(feature_map_conv1/BiasAdd/ReadVariableOp2R
'feature_map_conv1/Conv2D/ReadVariableOp'feature_map_conv1/Conv2D/ReadVariableOp2T
(feature_map_conv2/BiasAdd/ReadVariableOp(feature_map_conv2/BiasAdd/ReadVariableOp2R
'feature_map_conv2/Conv2D/ReadVariableOp'feature_map_conv2/Conv2D/ReadVariableOp2T
(feature_map_conv3/BiasAdd/ReadVariableOp(feature_map_conv3/BiasAdd/ReadVariableOp2R
'feature_map_conv3/Conv2D/ReadVariableOp'feature_map_conv3/Conv2D/ReadVariableOp2T
(feature_map_conv4/BiasAdd/ReadVariableOp(feature_map_conv4/BiasAdd/ReadVariableOp2R
'feature_map_conv4/Conv2D/ReadVariableOp'feature_map_conv4/Conv2D/ReadVariableOp2D
 feature_map_norm1/AssignNewValue feature_map_norm1/AssignNewValue2H
"feature_map_norm1/AssignNewValue_1"feature_map_norm1/AssignNewValue_12f
1feature_map_norm1/FusedBatchNormV3/ReadVariableOp1feature_map_norm1/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm1/FusedBatchNormV3/ReadVariableOp_13feature_map_norm1/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm1/ReadVariableOp feature_map_norm1/ReadVariableOp2H
"feature_map_norm1/ReadVariableOp_1"feature_map_norm1/ReadVariableOp_12D
 feature_map_norm2/AssignNewValue feature_map_norm2/AssignNewValue2H
"feature_map_norm2/AssignNewValue_1"feature_map_norm2/AssignNewValue_12f
1feature_map_norm2/FusedBatchNormV3/ReadVariableOp1feature_map_norm2/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm2/FusedBatchNormV3/ReadVariableOp_13feature_map_norm2/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm2/ReadVariableOp feature_map_norm2/ReadVariableOp2H
"feature_map_norm2/ReadVariableOp_1"feature_map_norm2/ReadVariableOp_12D
 feature_map_norm3/AssignNewValue feature_map_norm3/AssignNewValue2H
"feature_map_norm3/AssignNewValue_1"feature_map_norm3/AssignNewValue_12f
1feature_map_norm3/FusedBatchNormV3/ReadVariableOp1feature_map_norm3/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm3/FusedBatchNormV3/ReadVariableOp_13feature_map_norm3/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm3/ReadVariableOp feature_map_norm3/ReadVariableOp2H
"feature_map_norm3/ReadVariableOp_1"feature_map_norm3/ReadVariableOp_12D
 feature_map_norm4/AssignNewValue feature_map_norm4/AssignNewValue2H
"feature_map_norm4/AssignNewValue_1"feature_map_norm4/AssignNewValue_12f
1feature_map_norm4/FusedBatchNormV3/ReadVariableOp1feature_map_norm4/FusedBatchNormV3/ReadVariableOp2j
3feature_map_norm4/FusedBatchNormV3/ReadVariableOp_13feature_map_norm4/FusedBatchNormV3/ReadVariableOp_12D
 feature_map_norm4/ReadVariableOp feature_map_norm4/ReadVariableOp2H
"feature_map_norm4/ReadVariableOp_1"feature_map_norm4/ReadVariableOp_1:] Y
/
_output_shapes
:         
&
_user_specified_nameinput_images
б
╛
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_7186

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИвAssignNewValueвAssignNewValue_1вFusedBatchNormV3/ReadVariableOpв!FusedBatchNormV3/ReadVariableOp_1вReadVariableOpвReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype02
ReadVariableOp_1и
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype02!
FusedBatchNormV3/ReadVariableOpо
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1я
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,                           А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
╫#<2
FusedBatchNormV3┬
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue╬
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1К
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,                           А2

Identity▄
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,                           А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,                           А
 
_user_specified_nameinputs
╔
F
*__inference_digit_probs_layer_call_fn_9421

inputs
identity╞
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         
* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_digit_probs_layer_call_and_return_conditional_losses_76552
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         
:S O
+
_output_shapes
:         

 
_user_specified_nameinputs
К	
╩
map_while_cond_7451$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_7451___redundant_placeholder0:
6map_while_map_while_cond_7451___redundant_placeholder1
map_while_identity
В
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/LessМ
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
╡	
╦
0__inference_feature_map_norm3_layer_call_fn_9589

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИвStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_70602
StatefulPartitionedCallХ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*└
serving_defaultм
M
input_images=
serving_default_input_images:0         ?
digit_probs0
StatefulPartitionedCall:0         
tensorflow/serving/predict:╔Э
ї
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	optimizer
trainable_variables
	variables
	regularization_losses

	keras_api

signatures
Ж_default_save_signature
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_network
6
_init_input_shape"
_tf_keras_input_layer
 
	conv1
	norm1
	conv2
	norm2
	conv3
	norm3
	conv4
	norm4
trainable_variables
	variables
regularization_losses
	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
╦
	dconv
reshape

squash
trainable_variables
	variables
regularization_losses
	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
¤
 digit_caps_transform_tensor
 W
!digit_caps_log_priors
!B

"squash
#trainable_variables
$	variables
%regularization_losses
&	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"
_tf_keras_layer
з
'trainable_variables
(	variables
)regularization_losses
*	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"
_tf_keras_layer
у
+iter

,beta_1

-beta_2
	.decay
/learning_rate m▐!m▀0mр1mс2mт3mу4mф5mх6mц7mч8mш9mщ:mъ;mы<mь=mэ>mю?mя@mЁAmё vЄ!vє0vЇ1vї2vЎ3vў4v°5v∙6v·7v√8v№9v¤:v■;v <vА=vБ>vВ?vГ@vДAvЕ"
	optimizer
╢
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
@16
A17
 18
!19"
trackable_list_wrapper
Ў
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
B16
C17
D18
E19
F20
G21
H22
I23
@24
A25
 26
!27"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
Jmetrics
trainable_variables

Klayers
Llayer_regularization_losses
Mnon_trainable_variables
	variables
Nlayer_metrics
	regularization_losses
З__call__
Ж_default_save_signature
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
-
Сserving_default"
signature_map
 "
trackable_list_wrapper
╜

0kernel
1bias
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
Т__call__
+У&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
Saxis
	2gamma
3beta
Bmoving_mean
Cmoving_variance
Ttrainable_variables
U	variables
Vregularization_losses
W	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

4kernel
5bias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
\axis
	6gamma
7beta
Dmoving_mean
Emoving_variance
]trainable_variables
^	variables
_regularization_losses
`	keras_api
Ш__call__
+Щ&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

8kernel
9bias
atrainable_variables
b	variables
cregularization_losses
d	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
eaxis
	:gamma
;beta
Fmoving_mean
Gmoving_variance
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
Ь__call__
+Э&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

<kernel
=bias
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
Ю__call__
+Я&call_and_return_all_conditional_losses"
_tf_keras_layer
ь
naxis
	>gamma
?beta
Hmoving_mean
Imoving_variance
otrainable_variables
p	variables
qregularization_losses
r	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15"
trackable_list_wrapper
╓
00
11
22
33
44
55
66
77
88
99
:10
;11
<12
=13
>14
?15
B16
C17
D18
E19
F20
G21
H22
I23"
trackable_list_wrapper
 "
trackable_list_wrapper
░
smetrics
trainable_variables

tlayers
ulayer_regularization_losses
vnon_trainable_variables
	variables
wlayer_metrics
regularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
╜

@kernel
Abias
xtrainable_variables
y	variables
zregularization_losses
{	keras_api
в__call__
+г&call_and_return_all_conditional_losses"
_tf_keras_layer
з
|trainable_variables
}	variables
~regularization_losses
	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
л
Аtrainable_variables
Б	variables
Вregularization_losses
Г	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Дmetrics
trainable_variables
Еlayers
 Жlayer_regularization_losses
Зnon_trainable_variables
	variables
Иlayer_metrics
regularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
@:>
2&digit_caps/digit_caps_transform_tensor
6:4
2 digit_caps/digit_caps_log_priors
л
Йtrainable_variables
К	variables
Лregularization_losses
М	keras_api
и__call__
+й&call_and_return_all_conditional_losses"
_tf_keras_layer
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Нmetrics
#trainable_variables
Оlayers
 Пlayer_regularization_losses
Рnon_trainable_variables
$	variables
Сlayer_metrics
%regularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Тmetrics
'trainable_variables
Уlayers
 Фlayer_regularization_losses
Хnon_trainable_variables
(	variables
Цlayer_metrics
)regularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?:= 2%feature_maps/feature_map_conv1/kernel
1:/ 2#feature_maps/feature_map_conv1/bias
2:0 2$feature_maps/feature_map_norm1/gamma
1:/ 2#feature_maps/feature_map_norm1/beta
?:= @2%feature_maps/feature_map_conv2/kernel
1:/@2#feature_maps/feature_map_conv2/bias
2:0@2$feature_maps/feature_map_norm2/gamma
1:/@2#feature_maps/feature_map_norm2/beta
?:=@@2%feature_maps/feature_map_conv3/kernel
1:/@2#feature_maps/feature_map_conv3/bias
2:0@2$feature_maps/feature_map_norm3/gamma
1:/@2#feature_maps/feature_map_norm3/beta
@:>@А2%feature_maps/feature_map_conv4/kernel
2:0А2#feature_maps/feature_map_conv4/bias
3:1А2$feature_maps/feature_map_norm4/gamma
2:0А2#feature_maps/feature_map_norm4/beta
@:>		А2%primary_caps/primary_cap_dconv/kernel
2:0А2#primary_caps/primary_cap_dconv/bias
::8  (2*feature_maps/feature_map_norm1/moving_mean
>:<  (2.feature_maps/feature_map_norm1/moving_variance
::8@ (2*feature_maps/feature_map_norm2/moving_mean
>:<@ (2.feature_maps/feature_map_norm2/moving_variance
::8@ (2*feature_maps/feature_map_norm3/moving_mean
>:<@ (2.feature_maps/feature_map_norm3/moving_variance
;:9А (2*feature_maps/feature_map_norm4/moving_mean
?:=А (2.feature_maps/feature_map_norm4/moving_variance
0
Ч0
Ш1"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
X
B0
C1
D2
E3
F4
G5
H6
I7"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Щmetrics
Otrainable_variables
Ъlayers
 Ыlayer_regularization_losses
Ьnon_trainable_variables
P	variables
Эlayer_metrics
Qregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
<
20
31
B2
C3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
Юmetrics
Ttrainable_variables
Яlayers
 аlayer_regularization_losses
бnon_trainable_variables
U	variables
вlayer_metrics
Vregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
гmetrics
Xtrainable_variables
дlayers
 еlayer_regularization_losses
жnon_trainable_variables
Y	variables
зlayer_metrics
Zregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
<
60
71
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
иmetrics
]trainable_variables
йlayers
 кlayer_regularization_losses
лnon_trainable_variables
^	variables
мlayer_metrics
_regularization_losses
Ш__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
нmetrics
atrainable_variables
оlayers
 пlayer_regularization_losses
░non_trainable_variables
b	variables
▒layer_metrics
cregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
<
:0
;1
F2
G3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
▓metrics
ftrainable_variables
│layers
 ┤layer_regularization_losses
╡non_trainable_variables
g	variables
╢layer_metrics
hregularization_losses
Ь__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╖metrics
jtrainable_variables
╕layers
 ╣layer_regularization_losses
║non_trainable_variables
k	variables
╗layer_metrics
lregularization_losses
Ю__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
<
>0
?1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╝metrics
otrainable_variables
╜layers
 ╛layer_regularization_losses
┐non_trainable_variables
p	variables
└layer_metrics
qregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
B0
C1
D2
E3
F4
G5
H6
I7"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
╡
┴metrics
xtrainable_variables
┬layers
 ├layer_regularization_losses
─non_trainable_variables
y	variables
┼layer_metrics
zregularization_losses
в__call__
+г&call_and_return_all_conditional_losses
'г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╡
╞metrics
|trainable_variables
╟layers
 ╚layer_regularization_losses
╔non_trainable_variables
}	variables
╩layer_metrics
~regularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╦metrics
Аtrainable_variables
╠layers
 ═layer_regularization_losses
╬non_trainable_variables
Б	variables
╧layer_metrics
Вregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╨metrics
Йtrainable_variables
╤layers
 ╥layer_regularization_losses
╙non_trainable_variables
К	variables
╘layer_metrics
Лregularization_losses
и__call__
+й&call_and_return_all_conditional_losses
'й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

╒total

╓count
╫	variables
╪	keras_api"
_tf_keras_metric
c

┘total

┌count
█
_fn_kwargs
▄	variables
▌	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
╒0
╓1"
trackable_list_wrapper
.
╫	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
┘0
┌1"
trackable_list_wrapper
.
▄	variables"
_generic_user_object
@:>
2(digit_caps/digit_caps_transform_tensor/m
6:4
2"digit_caps/digit_caps_log_priors/m
?:= 2'feature_maps/feature_map_conv1/kernel/m
1:/ 2%feature_maps/feature_map_conv1/bias/m
2:0 2&feature_maps/feature_map_norm1/gamma/m
1:/ 2%feature_maps/feature_map_norm1/beta/m
?:= @2'feature_maps/feature_map_conv2/kernel/m
1:/@2%feature_maps/feature_map_conv2/bias/m
2:0@2&feature_maps/feature_map_norm2/gamma/m
1:/@2%feature_maps/feature_map_norm2/beta/m
?:=@@2'feature_maps/feature_map_conv3/kernel/m
1:/@2%feature_maps/feature_map_conv3/bias/m
2:0@2&feature_maps/feature_map_norm3/gamma/m
1:/@2%feature_maps/feature_map_norm3/beta/m
@:>@А2'feature_maps/feature_map_conv4/kernel/m
2:0А2%feature_maps/feature_map_conv4/bias/m
3:1А2&feature_maps/feature_map_norm4/gamma/m
2:0А2%feature_maps/feature_map_norm4/beta/m
@:>		А2'primary_caps/primary_cap_dconv/kernel/m
2:0А2%primary_caps/primary_cap_dconv/bias/m
@:>
2(digit_caps/digit_caps_transform_tensor/v
6:4
2"digit_caps/digit_caps_log_priors/v
?:= 2'feature_maps/feature_map_conv1/kernel/v
1:/ 2%feature_maps/feature_map_conv1/bias/v
2:0 2&feature_maps/feature_map_norm1/gamma/v
1:/ 2%feature_maps/feature_map_norm1/beta/v
?:= @2'feature_maps/feature_map_conv2/kernel/v
1:/@2%feature_maps/feature_map_conv2/bias/v
2:0@2&feature_maps/feature_map_norm2/gamma/v
1:/@2%feature_maps/feature_map_norm2/beta/v
?:=@@2'feature_maps/feature_map_conv3/kernel/v
1:/@2%feature_maps/feature_map_conv3/bias/v
2:0@2&feature_maps/feature_map_norm3/gamma/v
1:/@2%feature_maps/feature_map_norm3/beta/v
@:>@А2'feature_maps/feature_map_conv4/kernel/v
2:0А2%feature_maps/feature_map_conv4/bias/v
3:1А2&feature_maps/feature_map_norm4/gamma/v
2:0А2%feature_maps/feature_map_norm4/beta/v
@:>		А2'primary_caps/primary_cap_dconv/kernel/v
2:0А2%primary_caps/primary_cap_dconv/bias/v
╧B╠
__inference__wrapped_model_6742input_images"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
О2Л
0__inference_Efficient-CapsNet_layer_call_fn_7638
0__inference_Efficient-CapsNet_layer_call_fn_8396
0__inference_Efficient-CapsNet_layer_call_fn_8459
0__inference_Efficient-CapsNet_layer_call_fn_8130└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
·2ў
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8704
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8949
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8196
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8262└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Щ2Ц
+__inference_feature_maps_layer_call_fn_9002
+__inference_feature_maps_layer_call_fn_9055╣
░▓м
FullArgSpec/
args'Ъ$
jself
jinput_images

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╧2╠
F__inference_feature_maps_layer_call_and_return_conditional_losses_9143
F__inference_feature_maps_layer_call_and_return_conditional_losses_9231╣
░▓м
FullArgSpec/
args'Ъ$
jself
jinput_images

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪
+__inference_primary_caps_layer_call_fn_9240и
Я▓Ы
FullArgSpec#
argsЪ
jself
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
F__inference_primary_caps_layer_call_and_return_conditional_losses_9273и
Я▓Ы
FullArgSpec#
argsЪ
jself
jfeature_maps
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┘2╓
)__inference_digit_caps_layer_call_fn_9284и
Я▓Ы
FullArgSpec#
argsЪ
jself
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
D__inference_digit_caps_layer_call_and_return_conditional_losses_9411и
Я▓Ы
FullArgSpec#
argsЪ
jself
jprimary_caps
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
*__inference_digit_probs_layer_call_fn_9416
*__inference_digit_probs_layer_call_fn_9421└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
E__inference_digit_probs_layer_call_and_return_conditional_losses_9430
E__inference_digit_probs_layer_call_and_return_conditional_losses_9439└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬B╦
"__inference_signature_wrapper_8333input_images"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
0__inference_feature_map_norm1_layer_call_fn_9452
0__inference_feature_map_norm1_layer_call_fn_9465┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9483
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9501┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
0__inference_feature_map_norm2_layer_call_fn_9514
0__inference_feature_map_norm2_layer_call_fn_9527┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9545
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9563┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
0__inference_feature_map_norm3_layer_call_fn_9576
0__inference_feature_map_norm3_layer_call_fn_9589┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9607
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9625┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2Ы
0__inference_feature_map_norm4_layer_call_fn_9638
0__inference_feature_map_norm4_layer_call_fn_9651┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╘2╤
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9669
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9687┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
и2ев
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2ли
Я▓Ы
FullArgSpec#
argsЪ
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2ли
Я▓Ы
FullArgSpec#
argsЪ
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2ли
Я▓Ы
FullArgSpec#
argsЪ
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
о2ли
Я▓Ы
FullArgSpec#
argsЪ
jself
jinput_vector
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
	J
Const▐
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8196О0123BC4567DE89:;FG<=>?HI@A к!EвB
;в8
.К+
input_images         
p 

 
к "%в"
К
0         

Ъ ▐
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8262О0123BC4567DE89:;FG<=>?HI@A к!EвB
;в8
.К+
input_images         
p

 
к "%в"
К
0         

Ъ ╪
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8704И0123BC4567DE89:;FG<=>?HI@A к!?в<
5в2
(К%
inputs         
p 

 
к "%в"
К
0         

Ъ ╪
K__inference_Efficient-CapsNet_layer_call_and_return_conditional_losses_8949И0123BC4567DE89:;FG<=>?HI@A к!?в<
5в2
(К%
inputs         
p

 
к "%в"
К
0         

Ъ ╢
0__inference_Efficient-CapsNet_layer_call_fn_7638Б0123BC4567DE89:;FG<=>?HI@A к!EвB
;в8
.К+
input_images         
p 

 
к "К         
╢
0__inference_Efficient-CapsNet_layer_call_fn_8130Б0123BC4567DE89:;FG<=>?HI@A к!EвB
;в8
.К+
input_images         
p

 
к "К         
п
0__inference_Efficient-CapsNet_layer_call_fn_8396{0123BC4567DE89:;FG<=>?HI@A к!?в<
5в2
(К%
inputs         
p 

 
к "К         
п
0__inference_Efficient-CapsNet_layer_call_fn_8459{0123BC4567DE89:;FG<=>?HI@A к!?в<
5в2
(К%
inputs         
p

 
к "К         
╛
__inference__wrapped_model_6742Ъ0123BC4567DE89:;FG<=>?HI@A к!=в:
3в0
.К+
input_images         
к "9к6
4
digit_probs%К"
digit_probs         
┤
D__inference_digit_caps_layer_call_and_return_conditional_losses_9411l к!9в6
/в,
*К'
primary_caps         
к ")в&
К
0         

Ъ М
)__inference_digit_caps_layer_call_fn_9284_ к!9в6
/в,
*К'
primary_caps         
к "К         
н
E__inference_digit_probs_layer_call_and_return_conditional_losses_9430d;в8
1в.
$К!
inputs         


 
p 
к "%в"
К
0         

Ъ н
E__inference_digit_probs_layer_call_and_return_conditional_losses_9439d;в8
1в.
$К!
inputs         


 
p
к "%в"
К
0         

Ъ Е
*__inference_digit_probs_layer_call_fn_9416W;в8
1в.
$К!
inputs         


 
p 
к "К         
Е
*__inference_digit_probs_layer_call_fn_9421W;в8
1в.
$К!
inputs         


 
p
к "К         
ц
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9483Ц23BCMвJ
Cв@
:К7
inputs+                            
p 
к "?в<
5К2
0+                            
Ъ ц
K__inference_feature_map_norm1_layer_call_and_return_conditional_losses_9501Ц23BCMвJ
Cв@
:К7
inputs+                            
p
к "?в<
5К2
0+                            
Ъ ╛
0__inference_feature_map_norm1_layer_call_fn_9452Й23BCMвJ
Cв@
:К7
inputs+                            
p 
к "2К/+                            ╛
0__inference_feature_map_norm1_layer_call_fn_9465Й23BCMвJ
Cв@
:К7
inputs+                            
p
к "2К/+                            ц
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9545Ц67DEMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ц
K__inference_feature_map_norm2_layer_call_and_return_conditional_losses_9563Ц67DEMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ╛
0__inference_feature_map_norm2_layer_call_fn_9514Й67DEMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @╛
0__inference_feature_map_norm2_layer_call_fn_9527Й67DEMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @ц
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9607Ц:;FGMвJ
Cв@
:К7
inputs+                           @
p 
к "?в<
5К2
0+                           @
Ъ ц
K__inference_feature_map_norm3_layer_call_and_return_conditional_losses_9625Ц:;FGMвJ
Cв@
:К7
inputs+                           @
p
к "?в<
5К2
0+                           @
Ъ ╛
0__inference_feature_map_norm3_layer_call_fn_9576Й:;FGMвJ
Cв@
:К7
inputs+                           @
p 
к "2К/+                           @╛
0__inference_feature_map_norm3_layer_call_fn_9589Й:;FGMвJ
Cв@
:К7
inputs+                           @
p
к "2К/+                           @ш
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9669Ш>?HINвK
DвA
;К8
inputs,                           А
p 
к "@в=
6К3
0,                           А
Ъ ш
K__inference_feature_map_norm4_layer_call_and_return_conditional_losses_9687Ш>?HINвK
DвA
;К8
inputs,                           А
p
к "@в=
6К3
0,                           А
Ъ └
0__inference_feature_map_norm4_layer_call_fn_9638Л>?HINвK
DвA
;К8
inputs,                           А
p 
к "3К0,                           А└
0__inference_feature_map_norm4_layer_call_fn_9651Л>?HINвK
DвA
;К8
inputs,                           А
p
к "3К0,                           А╪
F__inference_feature_maps_layer_call_and_return_conditional_losses_9143Н0123BC4567DE89:;FG<=>?HIAв>
7в4
.К+
input_images         
p 
к ".в+
$К!
0         		А
Ъ ╪
F__inference_feature_maps_layer_call_and_return_conditional_losses_9231Н0123BC4567DE89:;FG<=>?HIAв>
7в4
.К+
input_images         
p
к ".в+
$К!
0         		А
Ъ ░
+__inference_feature_maps_layer_call_fn_9002А0123BC4567DE89:;FG<=>?HIAв>
7в4
.К+
input_images         
p 
к "!К         		А░
+__inference_feature_maps_layer_call_fn_9055А0123BC4567DE89:;FG<=>?HIAв>
7в4
.К+
input_images         
p
к "!К         		А╣
F__inference_primary_caps_layer_call_and_return_conditional_losses_9273o@A>в;
4в1
/К,
feature_maps         		А
к ")в&
К
0         
Ъ С
+__inference_primary_caps_layer_call_fn_9240b@A>в;
4в1
/К,
feature_maps         		А
к "К         ╤
"__inference_signature_wrapper_8333к0123BC4567DE89:;FG<=>?HI@A к!MвJ
в 
Cк@
>
input_images.К+
input_images         "9к6
4
digit_probs%К"
digit_probs         
