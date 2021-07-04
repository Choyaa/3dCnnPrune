import os
f= open("vi.txt","r")
data = f.read()
b = list(data)
cnt = 0
for i in range(len(data)):
    if data[i] == 't':
        b.insert(i+cnt,'\r\n')
        cnt += 1
a_b = ''.join(b)
ff = open("vv.txt","w")
print(a_b,file = ff)
ff.close
