"""
A long list containing reference dartboard corner points for a 1152x1152 pixel img
"""
import numpy as np

board_corners_reference = np.asarray([

    (583, 530),  # bull, 20c4, 1c1
    (620.33, 293.33),  # triple, 20c4, 1c1
    (624, 271),  # triple, 20c3, 1c2
    (648.5, 114.5),  # double, 20c4, 1c1
    (651.67, 91.33),  # double, 20c3, 1c2

    (596.67, 534.33),  # bull, 1c4, 18c1
    (705, 321),  # triple, 1c4, 18c1
    (715.33, 300.33),  # triple, 1c3, 18c2
    (787.5, 159.5),  # double, 1c4, 18c1
    (798, 139),  # double, 1c3, 18c2

    (608, 543),  # bull, 18c4, 4c1
    (777.33, 373.33),  # triple, 18c4, 4c1
    (793.5, 357.5),  # triple, 18c3, 4c2
    (906, 245),  # double, 18c4, 4c1
    (922, 229),  # double, 18c3, 4c2

    (616.67, 554.33),  # bull, 4c4, 13c1
    (830, 446),  # triple, 4c4, 13c1
    (850.67, 435.33),  # triple, 4c3, 13c2
    (991.67, 363.33),  # double, 4c4, 13c1
    (1012, 353),  # double, 4c3, 13c2

    (621, 568),  # bull, 13c4, 6c1
    (857.33, 530.67),  # triple, 13c4, 6c1
    (880.67, 527.33),  # triple, 13c3, 6c2
    (1036.67, 502.33),  # double, 13c4, 6c1
    (1059.67, 498.67),  # double, 13c3, 6c2

    (621, 582),  # bull, 6c4, 10c1
    (857, 620),  # triple, 6c4, 10c1
    (880, 623),  # triple 6c3, 10c2
    (1036.67, 648.67),  # double 6c4, 10c1
    (1059, 652),  # double, 6c3, 10c2

    (616.67, 596.33),  # bull, 10c4, 15c1
    (830, 705),  # triple, 10c4, 15c1
    (850.33, 715.67),  # triple, 10c3, 15c2
    (991.5, 787.5),  # double, 10c4, 15c1
    (1012, 798),  # double, 10c3, 15c2

    (608, 608),  # bull, 15c4, 2c1
    (777.5, 777.5),  # triple, 15c4, 2c1
    (793.5, 793.5),  # triple 15c3, 2c2
    (906, 906),  # double, 15c4, 2c1
    (922, 922),  # double, 15c3, 2c2

    (596.33, 616.67),  # bull, 2c4, 17c1
    (705, 830),  # triple, 2c4, 17c1
    (715.33, 850.33),  # triple, 2c3, 17c2
    (787.5, 991.5),  # double, 2c4, 17c1
    (798, 1012),  # double, 2c3, 17c2

    (583, 621),  # bull, 17c4, 3c1
    (620.33, 857.33),  # triple, 17c4, 3c1
    (624, 880),  # triple, 17c3, 3c2
    (648.5, 1036.5),  # double, 17c4, 3c1
    (652, 1059),  # double, 17c3, 3c2

    (568, 621),  # bull, 3c4, 19c1
    (530.67, 857.67),  # triple, 3c4, 19c1
    (527, 880),  # triple, 3c3, 19c2
    (502.5, 1036.5),  # double, 3c4, 19c1
    (499, 1059),  # double, 3c3, 19c2

    (554.5, 616.5),  # bull, 19c4, 7c1
    (446, 830),  # triple, 19c4, 7c1
    (435.33, 850.33),  # triple, 19c3, 7c2
    (363.5, 991.5),  # double, 19c4, 7c1
    (353, 1012),  # double, 19c3, 7c2

    (543, 608),  # bull, 7c4, 16c1
    (374, 777),  # triple, 7c4, 16c1
    (357, 794),  # triple, 7c3, 16c2
    (245, 906),  # double, 7c4, 16c1
    (229, 922),  # double, 7c3, 16c2

    (534.5, 596.5),  # bull, 16c4, 8c1
    (321, 705),  # triple, 16c4, 8c1
    (300.67, 715.67),  # triple 16c3, 8c2
    (159.5, 787.5),  # double, 16c4, 8c1
    (139, 798),  # double, 16c3, 8c2

    (530, 583),  # bull, 8c4, 11c1
    (293.67, 620.33),  # triple, 8c4, 11c1
    (270.33, 623.67),  # triple, 8c3, 11c2
    (114.5, 648.5),  # double 8c4, 11c1
    (92, 652),  # double 8c3, 11c2

    (530.33, 568.67),  # bull, 11c4, 14c1
    (293, 531),  # triple, 11c4, 14c1
    (270.67, 527.67),  # triple, 11c3, 14c2
    (114.5, 502.5),  # double, 11c4, 14c1
    (91, 499),  # double, 11c3, 14c2

    (534.33, 554.67),  # bull, 14c4, 9c1
    (321, 446),  # triple, 14c4, 9c1
    (300.67, 435.33),  # triple, 14c3, 9c2
    (159.5, 363.5),  # double, 14c4, 9c1
    (139, 353),  # double, 14c3, 9c2

    (542.67, 542.67),  # bull, 9c4, 12c1
    (374, 374),  # triple, 9c4, 12c1
    (357.5, 357.5),  # triple, 9c3, 12c2
    (245, 245),  # double, 9c4, 12c1
    (229, 229),  # double, 9c3, 12c2

    (555, 534),  # bull, 12c4, 5c1
    (446, 321),  # triple, 12c4, 5c1
    (435.5, 300.5),  # triple, 12c3, 5c2
    (363.67, 159.33),  # double, 12c4, 5c1
    (353, 139),  # double, 12c3, 5c2

    (568, 530),  # bull, 5c1, 20c1
    (531, 293),  # triple, 5c4, 20c1
    (527.5, 270.5),  # triple, 5c3, 20c2
    (502.5, 114.5),  # double, 5c4, 20c1
    (499.5, 91.5),  # double, 5c3, 20c2
])

board_corners_reference_backup = [
    (652, 1059),  # double, 17c3, 3c2
    (499, 1059),  # double, 3c3, 19c2
    (648.5, 1036.5),  # double, 17c4, 3c1
    (502.5, 1036.5),  # double, 3c4, 19c1
    (798, 1012),  # double, 2c3, 17c2
    (353, 1012),  # double, 19c3, 7c2
    (787.5, 991.5),  # double, 2c4, 17c1
    (363.5, 991.5),  # double, 19c4, 7c1
    (922, 922),  # double, 15c3, 2c2
    (229, 922),  # double, 7c3, 16c2
    (906, 906),  # double, 15c4, 2c1
    (245, 906),  # double, 7c4, 16c1
    (624, 880),  # triple, 17c3, 3c2
    (527, 880),  # triple, 3c3, 19c2
    (620.33, 857.33),  # triple, 17c4, 3c1
    (530.67, 857.67),  # triple, 3c4, 19c1
    (715.33, 850.33),  # triple, 2c3, 17c2
    (435.33, 850.33),  # triple, 19c3, 7c2
    (705, 830),  # triple, 2c4, 17c1
    (446, 830),  # triple, 19c4, 7c1
    (1012, 798),  # double, 10c3, 15c2
    (139, 798),  # double, 16c3, 8c2
    (357, 794),  # triple, 7c3, 16c2
    (793.5, 793.5),  # triple 15c3, 2c2
    (991.5, 787.5),  # double, 10c4, 15c1
    (159.5, 787.5),  # double, 16c4, 8c1
    (777.5, 777.5),  # triple, 15c4, 2c1
    (374, 777),  # triple, 7c4, 16c1
    (850.33, 715.67),  # triple, 10c3, 15c2
    (300.67, 715.67),  # triple 16c3, 8c2
    (830, 705),  # triple, 10c4, 15c1
    (321, 705),  # triple, 16c4, 8c1
    (1059, 652),  # double, 6c3, 10c2
    (92, 652),  # double 8c3, 11c2
    (1036.67, 648.67),  # double 6c4, 10c1
    (114.5, 648.5),  # double 8c4, 11c1
    (880, 623),  # triple 6c3, 10c2
    (270.33, 623.67),  # triple, 8c3, 11c2
    (583, 621),  # bull, 17c4, 3c1
    (568, 621),  # bull, 3c4, 19c1
    (857, 620),  # triple, 6c4, 10c1
    (293.67, 620.33),  # triple, 8c4, 11c1
    (596.33, 616.67),  # bull, 2c4, 17c1
    (554.5, 616.5),  # bull, 19c4, 7c1
    (608, 608),  # bull, 15c4, 2c1
    (543, 608),  # bull, 7c4, 16c1
    (616.67, 596.33),  # bull, 10c4, 15c1
    (534.5, 596.5),  # bull, 16c4, 8c1
    (530, 583),  # bull, 8c4, 11c1
    (621, 582),  # bull, 6c4, 10c1
    (621, 568),  # bull, 13c4, 6c1
    (530.33, 568.67),  # bull, 11c4, 14c1
    (616.67, 554.33),  # bull, 4c4, 13c1
    (534.33, 554.67),  # bull, 14c4, 9c1
    (608, 543),  # bull, 18c4, 4c1
    (542.67, 542.67),  # bull, 9c4, 12c1
    (596.67, 534.33),  # bull, 1c4, 18c1
    (555, 534),  # bull, 12c4, 5c1
    (293, 531),  # triple, 11c4, 14c1
    (857.33, 530.67),  # triple, 13c4, 6c1
    (583, 530),  # bull, 20c4, 1c1
    (568, 530),  # bull, 5c1, 20c1
    (880.67, 527.33),  # triple, 13c3, 6c2
    (270.67, 527.67),  # triple, 11c3, 14c2
    (1036.67, 502.33),  # double, 13c4, 6c1
    (114.5, 502.5),  # double, 11c4, 14c1
    (91, 499),  # double, 11c3, 14c2
    (1059.67, 498.67),  # double, 13c3, 6c2
    (830, 446),  # triple, 4c4, 13c1
    (321, 446),  # triple, 14c4, 9c1
    (850.67, 435.33),  # triple, 4c3, 13c2
    (300.67, 435.33),  # triple, 14c3, 9c2
    (374, 374),  # triple, 9c4, 12c1
    (777.33, 373.33),  # triple, 18c4, 4c1
    (991.67, 363.33),  # double, 4c4, 13c1
    (159.5, 363.5),  # double, 14c4, 9c1
    (793.5, 357.5),  # triple, 18c3, 4c2
    (357.5, 357.5),  # triple, 9c3, 12c2
    (1012, 353),  # double, 4c3, 13c2
    (139, 353),  # double, 14c3, 9c2
    (705, 321),  # triple, 1c4, 18c1
    (446, 321),  # triple, 12c4, 5c1
    (715.33, 300.33),  # triple, 1c3, 18c2
    (435.5, 300.5),  # triple, 12c3, 5c2
    (620.33, 293.33),  # triple, 20c4, 1c1
    (531, 293),  # triple, 5c4, 20c1
    (624, 271),  # triple, 20c3, 1c2
    (527.5, 270.5),  # triple, 5c3, 20c2
    (906, 245),  # double, 18c4, 4c1
    (245, 245),  # double, 9c4, 12c1
    (922, 229),  # double, 18c3, 4c2
    (229, 229),  # double, 9c3, 12c2
    (787.5, 159.5),  # double, 1c4, 18c1
    (363.67, 159.33),  # double, 12c4, 5c1
    (798, 139),  # double, 1c3, 18c2
    (353, 139),  # double, 12c3, 5c2
    (648.5, 114.5),  # double, 20c4, 1c1
    (502.5, 114.5),  # double, 5c4, 20c1
    (651.67, 91.33),  # double, 20c3, 1c2
    (499.5, 91.5)  # double, 5c3, 20c2
]

