
# i = 1

for i in range(1, 26):
    f = open('email/ham/%d.txt' % i).read()

    print(f, i)

# i = 23
#
# f = open('email/ham/%d.txt' % i).read()
#
# print(f, i)