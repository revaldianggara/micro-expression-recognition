from PIL import Image

img = Image.open('D:/micro-expresion/Aplikasi Baru/Aplikasi2.2/capture/1.jpg')
imgWidth, imgHeight = img.size
img = img.convert("RGBA")
imgdata = img.getdata()

x_pos = 0
y_pos = 1

pixel_value = []
x = []
y = []

for item in imgdata:
    if (x_pos) == imgWidth:
        x_pos = 1
        y_pos += 1
    else:
        x_pos += 1

    if item[3] != 0:
        pixel_value.append(item[2])
        x.append(x_pos)
        y.append(y_pos)

pixel_value, x, y = zip(*sorted(zip(pixel_value, x, y), reverse=True))

print(f'{pixel_value}\n{x}\n{y}')