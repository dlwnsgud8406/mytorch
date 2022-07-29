file = open('neural_network.txt')
line = file.readline()
first=[]
second = []
third = []

while line:
    line = file.readline()
    if ':  ' in line:
        prediction = line.split(':  ')[1]
        if ',  ' in prediction :
            first.append(float(prediction.split(', ')[0]))
            second.append(float(prediction.split(', ')[1]))
            third.append(float(prediction.split(', ')[2]))
first_sum, second_sum, third_sum = 0, 0, 0

for i in range(0, len(first)):
    first_sum += first[i]

for i in range(0, len(second)):
    second_sum += second[i]

for i in range(0, len(third)):
    third_sum += third[i]

first_average = first_sum / len(first)
second_average = second_sum / len(second)
third_average = third_sum / len(third)

print(first_average, second_average, third_average)
