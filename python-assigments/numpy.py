import numpy as np
# 1-create an 1D array of numbers from  10 to 21
array1d = np.arange(10,22)
print("1D array of numbers from  10 to 21",array1d)
# 2-create an array of floats and convert it to integers
floatArray = np.array([0.9,10.222,4.56,6.98,3.44])
print("Floats to Integers",floatArray.astype(int))
# 3-create a 2D array of shape (3,4) & extract second row & first column
array2D = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
second_row = array2D[1, :]
first_column = array2D[:, 0]
print("Array 2D",array2D)
print("Second row",second_row)
print("first column",first_column)

# 4-create a 2 arrays of size 5 and perform multi,add,sub,div
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10])
multip = array1 * array2   
add = array1 + array2         
sub= array1 - array2      
div= array1 / array2         
print("Array 1:", array1)
print("Array 2:", array2)
print("Multiplication:", multip)
print("Addition:", add)
print("Subtraction:", sub)
print("Division:", div)

# 5-create a 3D array using np.ones reshape it into (2,3,4)
array = np.ones(24)
reshaped_array = array.reshape(2, 3, 4)
print("Reshaped 3D array:")
print(reshaped_array)
# 6-create an array of numbers from 1 to 50 & filtered out  numbers greater than 15
array = np.arange(1, 51)
filtered_array = array[array > 15]
print("Filtered numbers greater than 15):",filtered_array)
# 7-create an array & make a deep copy modify it  & show that original one stay unchannged
original_array = np.array([1,2,3,4,5,6])
deep_copy = original_array.copy()
deep_copy[0] = 100
print("Original array:", original_array)
print("Deep copy:", deep_copy)

# 8-create 1D arrya & sort it in ascending & descending
array_AD = np.array([5, 2, 9, 1, 7, 3])
ascending_array = np.sort(array_AD)
descending_array = np.sort(array_AD)[::-1]
print("Original array:", array_AD)
print("Ascending order:", ascending_array)
print("Descending order:", descending_array)

# 9-create a 4X4 array & extract  subarray last two rows & first two columns
array_4by4 = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
subarray = array_4by4[2:, :2]
print("Original array:",array_4by4)
print("Subarray (last two rows, first two columns):",subarray)


# 10-create an array of random integers between 1-20 & filter out even numbers(np.random.randit)
randomArray = np.random.randint(1, 21, size=10)
filtered_array = randomArray[randomArray % 2 != 0]
print("Original array:", randomArray)
print("Filtered array (odd numbers only):", filtered_array)

# 11-create an 1D array and calculate the square root and exponentiate of each elemrnt(np.sqrt)(np.exp)
array = np.array([1, 4, 9, 16, 25])
square_root = np.sqrt(array)
exponentiation = np.exp(array)
print("Original array:", array)
print("Square root of each element:", square_root)
print("Exponentiate of each element:", exponentiation)
