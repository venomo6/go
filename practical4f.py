from PIL import Image
im = Image.open('cat.jpg')
cmyk= im.convert('CMYK').split()
c = cmyk[0].convert('1').convert('L')
m = cmyk[1].convert('1').convert('L')
y = cmyk[2].convert('1').convert('L')
k = cmyk[3].convert('1').convert('L')
new_cmyk = Image.merge('CMYK',[c,m,y,k])
new_cmyk.show()
new_cmyk.save('Halftoaning.jpg')
print('Halftoaning Done!')
