import time
import random

r = lambda *args: random.choice(args)

def back_up_data(data, length = 50):
    case = []

    for x in range(length):
        right = [r(-1, 0), r(-1, 0)] 
        left = [r(-1, 0), r(-1, 0)]
        distance = right + [r(1, 0.9) for i in range(4)] + left
        cam = [0] * 5

        wheels = [-1, 1]
        if left == right:
            wheels = [-1,-0.6]
        elif left == [-1, -1]:
            wheels = [1, -1]

        case.append([distance + cam, wheels])

    data.extend(case)

def sway (data, length = 60):
    
    case = []

    for x in range(length/2):
        distance = [.9] + [r(0, -1)] + [r(-0.3, -1)] + [.9]*3 + [.9]*2
        cam = [0] * 5
        case.append([distance + cam, [-1, 1]])

    for x in range(length/2):
        distance = [0.9]*2 + [0.9]*3 + [r(-0.3, -1)] + [r(0, -1)] + [0.9]
        cam = [0] * 5
        case.append([distance + cam, [1, -1]])

    data.extend(case)

def turn_left_data(data, length = 50):

    case = []

    for x in range(length):
        distance = [r(-1, 0.3, 0), r(-1, 0.3, 0)] + [r(0, 0.5, 1) for i in range(4)] + [r(0.9, 1.0), r(0.9, 1.0)]
        cam = [r(0, 0.3) for i in range(5)]
        case.append([distance + cam, [-1,1.0]])

    data.extend(case)

def turn_right_data(data, length = 50):

    case = []

    for x in range(length):
        distance = [r(0.9, 1.0), r(0.9, 1.0)] + [r(0, 0.5, 1) for i in range(4)] + [r(-1, 0), r(-1,  0)]
        cam = [r(0, 0.3) for i in range(5)]
        case.append([distance + cam, [1.0,-1]])

    data.extend(case)


def stop_data(data, length = 100):

    case = []

    for x in range(length):
        # distance = [r(0.9, 1.0), r(0.9, 1.0)] + [r(0, 0.5) for i in range(4)] + [r(0.9, 1.0), r(0.9, 1.0)]
        distance = [r(0, 0.3, -.5, -1, 1) for i in range(8)]
        cam = [r(0.9, 1, 0.6) for i in range(5)]
        case.append([distance + cam, [0,0]])

    data.extend(case)

def move_forward_data(data, length=400):

    case = [1]*(8) + [0]*5
    data.extend([[case, [1, 1]] for i in range(length)])

    case = [1]*(8) + [1, 1] + [.1]*3
    data.extend([[case, [-.5, .5]] for i in range(length)])

    case = [1]*(8) + [.1]*3 + [1, 1]
    data.extend([[case, [.5, -.5]] for i in range(length)])
# for i in range(length/2):
#         i = r(1, 0)
#         o = r(1-i, 0, 1)
#         case = [1]*(8) + [o, i] + [.1]*3

#         if not(i == o == 0):
#             data.append([case, [0, .5]])
        
#     for i in range(length/2):
#         i = r(1, 0)
#         o = r(1-i, 0, 1)
#         case = [1]*(8) + [.1]*3 + [i, o]

#         if not(i == o == 0):
#             data.append([case, [.5, 0]])

def fill_learning_data(filename, data):

    print time.time()
    with open(filename, 'wr') as open_file:
        for target in data:
            open_file.write("%s\n" % str(target))
        

data = []
back_up_data(data)
turn_left_data(data)
turn_right_data(data)
sway(data)
stop_data(data, 100)
move_forward_data(data)

# find_object(data)

fill_learning_data("data/test.txt", data)