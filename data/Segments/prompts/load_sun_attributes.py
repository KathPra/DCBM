## conda env_clipeval

import scipy.io
mat = scipy.io.loadmat('prompts/SUNAttributeDB/attributes.mat')
sun_attributes = mat['attributes']
for i in range(len(sun_attributes)):
    sun_attributes[i] = str(sun_attributes[i][0][0])
print(sun_attributes)

with open("prompts/sun_attributes.txt", "w") as f:
    for i in sun_attributes:
        f.write(i[0]+"\n")