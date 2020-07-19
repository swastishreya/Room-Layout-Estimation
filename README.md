# Room-Layout-Estimation
General Room Layout Estimation - one needs to reconstruct the enclosing structure of the indoor scene, consisting of walls, floor, and ceiling. In this challenge, we do not make any assumption on the room structure, such as cuboid-shaped or Manhattan layouts.

## Training
BCE + dice loss (Unet):

Epoch 0:train: bce: 0.806151, dice: 0.999955, loss: 0.903053
        val: bce: 0.789308, dice: 0.999950, loss: 0.894629

Epoch 4:train: bce: 0.752415, dice: 0.999955, loss: 0.876185
        val: bce: 0.746951, dice: 0.999950, loss: 0.873451

Epoch 9:train: bce: 0.702170, dice: 0.999955, loss: 0.851063
        val: bce: 0.697118, dice: 0.999950, loss: 0.848534
        
BCE (MajdoorNet):

Best model - train: bce: 0.066578, dice: 0.975578, loss: 0.066578
             val: bce: 0.066103, dice: 0.973867, loss: 0.066103
             

