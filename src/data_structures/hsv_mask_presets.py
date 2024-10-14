import abc
from dataclasses import dataclass
from typing import Union


@dataclass
class FilterRange:
    min: int
    max: int

    def get_filter_range(self) -> list:
        return list(range(self.min, self.max))


@dataclass
class HSVMask:
    # hue values can stay the same, the other values are more important filters
    hue_lower_1: int
    hue_upper_1: int
    sat_lower: int
    sat_upper: int
    val_lower: int
    val_upper: int
    hue_lower_2: Union[int, None] = None
    hue_upper_2: Union[int, None] = None

    def has_split_hue_values(self):
        return self.hue_lower_2 is not None and self.hue_lower_2 is not None

    def get_lower_1(self):
        return [self.hue_lower_1, self.sat_lower, self.val_lower]

    def get_upper_1(self):
        return [self.hue_upper_1, self.sat_upper, self.val_upper]

    def get_lower_2(self):
        return [self.hue_lower_2, self.sat_lower, self.val_lower]

    def get_upper_2(self):
        return [self.hue_upper_2, self.sat_upper, self.val_upper]


# red hue values are split from 0 to 9 and 150 to 179
hsv_mask_red = HSVMask(0, 9, 85, 255, 32, 255, 150, 179)
hsv_mask_green = HSVMask(34, 85, 30, 255, 35, 255)
hsv_mask_silver = HSVMask(0, 179, 0, 84, 97, 255)

    # def get_all_mask_variations(self):
    #     if isinstance(self.sat_lower, FilterRange):
    #         range_sat_lower = self.sat_lower.get_filter_range()
    #     else:
    #         range_sat_lower = [self.sat_lower]
    #     if isinstance(self.sat_upper, FilterRange):
    #         range_sat_upper = self.sat_upper.get_filter_range()
    #     else:
    #         range_sat_upper = [self.sat_upper]
    #     if isinstance(self.val_lower, FilterRange):
    #         range_val_lower = self.val_lower.get_filter_range()
    #     else:
    #         range_val_lower = [self.val_lower]
    #
    #     if isinstance(self.val_upper, FilterRange):
    #         range_val_upper = self.val_upper.get_filter_range()
    #     else:
    #         range_val_upper = [self.val_upper]
    #
    #     hsv_mask_variations = [
    #         HSVMask(self.hue_lower_1, self.hue_upper_1, sl, su, vl, vu, self.hue_lower_2, self.hue_upper_2)
    #         for sl in range_sat_lower
    #         for su in range_sat_upper
    #         for vl in range_val_lower
    #         for vu in range_val_upper
    #     ]
    #
    #     return hsv_mask_variations

#
# @dataclass
# class HSVMaskRedRange(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 9
#         self.sat_lower = FilterRange(70, 110)
#         self.sat_upper = 255
#         self.val_lower = 32
#         self.val_upper = 255
#         self.hue_lower_2 = 150
#         self.hue_upper_2 = 179
#
#
# @dataclass
# class HSVMaskGreenRange(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 34
#         self.hue_upper_1 = 85
#         self.sat_lower = FilterRange(40, 70)
#         self.sat_upper = 255
#         self.val_lower = FilterRange(30, 40)
#         self.val_upper = 255
#
#
# @dataclass
# class HSVMaskWhiteRange(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 36
#         self.sat_lower = 0
#         self.sat_upper = FilterRange(70, 160)
#         self.val_lower = FilterRange(11, 140)
#         self.val_upper = 255
#         self.hue_lower_2 = 100
#         self.hue_upper_2 = 179
#
#
# @dataclass
# class HSVMaskBlackRange(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 179
#         self.sat_lower = 0
#         self.sat_upper = 255
#         self.val_lower = 0
#         self.val_upper = FilterRange(40, 70)
#



#
# @dataclass
# class HSVMaskRedEq(HSVMask):
#
#     hue_lower_1 = 0
#     hue_upper_1 = 9
#     sat_lower = 85
#     sat_upper = 255
#     val_lower = 32
#     val_upper = 255
#     hue_lower_2 = 150
#     hue_upper_2 = 179
#
#
# @dataclass
# class HSVMaskGreenEq(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 34
#         self.hue_upper_1 = 85
#         self.sat_lower = 30
#         self.sat_upper = 255
#         self.val_lower = 35
#         self.val_upper = 255

#
# @dataclass
# class HSVMaskSilverEQ(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 179
#         self.sat_lower = 0
#         self.sat_upper = 84
#         self.val_lower = 97
#         self.val_upper = 255


# @dataclass
# class HSVMaskWhiteEq(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 36
#         self.sat_lower = 0
#         self.sat_upper = 102
#         self.val_lower = 89
#         self.val_upper = 255
#         self.hue_lower_2 = 100
#         self.hue_upper_2 = 179
#
#
# @dataclass
# class HSVMaskBlackEq(HSVMask):
#
#     def __post_init__(self):
#         self.hue_lower_1 = 0
#         self.hue_upper_1 = 179
#         self.sat_lower = 0
#         self.sat_upper = 255
#         self.val_lower = 0
#         self.val_upper = 88
#
#
# @dataclass(frozen=True)
# class HSVMaskCollection:
#     standard: HSVMask
#     backups: [HSVMask]
#
#
# red_masks = HSVMaskCollection(
#     standard=HSVMask(
#         hue_lower_1=0,
#         hue_upper_1=9,
#         sat_lower=FilterRange(70, 255),
#         sat_upper=255,
#         val_lower=32,
#         val_upper=255,
#         hue_lower_2=150,
#         hue_upper_2=179
#     ),
#     backups=[
#         HSVMask(0, 9, 50, 255, 105, 255, 150, 179),
#         HSVMask(0, 8, 127, 255, 0, 255, 150, 179),
#         HSVMask(0, 8, 86, 255, 126, 255, 150, 179)
#     ]
# )
#
# green_masks = HSVMaskCollection(
#     standard=HSVMask(63, 80, 48, 255, 62, 255),
#     backups=[
#         HSVMask(37, 94, 32, 255, 40, 255),
#         HSVMask(37, 86, 63, 255, 60, 255)
#     ]
# )
#
# white_masks = HSVMaskCollection(
#     standard=HSVMask(0, 179, 0, 95, 93, 255),
#     backups=[
#         HSVMask(0, 179, 0, 117, 138, 255),
#         HSVMask(0, 179, 0, 126, 140, 255),
#         HSVMask(0, 64, 0, 81, 89, 255),
#         HSVMask(0, 36, 12, 130, 121, 255)
#     ]
# )
#
# black_masks = HSVMaskCollection(
#     standard=HSVMask(0, 179, 0, 255, 0, 61),
#     backups=[HSVMask(0, 179, 0, 155, 0, 76)]
# )
#
#
# silver_masks = HSVMaskCollection(
#     standard=HSVMask(0, 179, 0, 86, 57, 122),
#     backups=[]
# )

# red_and_green_masks = HSVMaskCollection(
#     standard=HSVMask(0, 179, 120, 255, 86, 255),
#     backups=[]
# )
