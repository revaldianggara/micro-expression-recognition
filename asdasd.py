

rows = 13
print("*" * rows, end="\n")
# pyramid height: 6
i = (rows // 2) - 1
j = 2
while i != 0:
    while j <= (rows - 2):
        print("." * i, end="")
        print("*" * j, end="")
        print("." * i, end="\n")
        i = i - 1
        j = j + 2

    print('............')
    print('FINISH \(^_^)/ ')