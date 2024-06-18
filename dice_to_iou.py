
'''
def dice_to_iou(dice):
    return dice / (2 - dice)

dice = 
iou = dice_to_iou(dice)
print(f"IoU: {iou:.4f}")
'''
def iou_to_dice(iou):
    return (2 * iou) / (1 + iou)

iou = 0.4609

dice = iou_to_dice(iou)
print(f"Dice: {dice:.4f}")
