import matplotlib.pyplot as plt
import random as rnd

PALETTES = {
    "Default": ["#dd3333","#33dd33","#3333dd"],
    "Artemis": ["#59bff0","#1e0e8c", "#3438dc", "#9125d5", "#c34dd9", "#b42b4c"],
    "Campfire": ["#638a7e", "#aebb8f", "#e7d192", "#e8b67f", "#cc7161"],
    "Demeter": ["#d23e32", "#79e5ef", "#26545d", "#6d1915"],
    "Grayscale": ["#000000", "#333333", "#666666", "#999999", "#bbbbbb", "#dddddd"],
    "Random": ["#000000"]
}

def rgb_to_hex(rgb):
    hex = '#%02x%02x%02x' % tuple(rgb)
    print("NewColor: ", hex)
    return hex
def hex_to_rgb(hex_code):
    hex_code = hex_code[1:]
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]
def colorswitch(hex):
    rgb = hex_to_rgb(hex)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    avg = (r+g+b)/3

    if avg > 127.5:
        for i, val in enumerate(rgb):

            if rnd.randint(1,10) > 5:
                shift = rnd.random()*0.5
                print(shift)
            else:
                shift = 0

            if val-(val*(0.2+shift)) > 0.0:
                rgb[i] -= round(rgb[i]*(0.2 + shift))

        return rgb_to_hex(rgb)
    else:
        for i, val in enumerate(rgb):

            if rnd.randint(1,10) > 5:
                shift = rnd.random()
            else:
                shift = 0
            
            if val+(255*(0.2+shift)) < 255.0:
                rgb[i] += round(255*(0.2+shift))
        return rgb_to_hex(rgb)

def simple_plot(datas=[['','']], title='', labels=['',''], legends=[''], show=True, palette="Default", linewidths=[]):
    
    pl = PALETTES[palette]
    diff = len(datas)-len(pl)
    if(diff):
        for i in range(diff):
            pl.append(colorswitch(pl[i]))

    for i, d in enumerate(datas):
        plt.plot(d[0], d[1], label=legends[i], color=pl[i])
    plt.title(title)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    if show:
        plt.show()

dev_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
dev_y = [ rnd.random()*60 for _ in range(len(dev_x))]

dev_x2 = [1, 2, 3, 4, 5, 10, 15, 20]
dev_y2 = [ rnd.random()*60 for _ in range(len(dev_x2))]

dev_x3 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 15, 20]
dev_y3 = [ rnd.randint(21, 30) for _ in range(len(dev_x3))]
dev_y4 = [ rnd.randint(31, 40) for _ in range(len(dev_x3))]
dev_y5 = [ rnd.randint(41, 50) for _ in range(len(dev_x3))]
dev_y6 = [ rnd.randint(51, 60) for _ in range(len(dev_x3))]

simple_plot(
    datas=[[dev_x,dev_y], [dev_x2,dev_y2], [dev_x3, dev_y3], [dev_x3, dev_y4], [dev_x3, dev_y5], [dev_x3, dev_y6]], 
    title="Titolo", 
    labels=["Dettaglio X", "Dettaglio Y"], 
    legends=['Dataset 1','Dataset 2','Dataset 3','Dataset 4','Dataset 5','Dataset 6'], 
    palette="Campfire",
    show=True
)