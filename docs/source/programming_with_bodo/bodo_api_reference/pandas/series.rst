.. _series:

Series
~~~~~~

Bodo provides extensive Series support.
However, operations between Series (+, -, /, *, **) do not
implicitly align values based on their
associated index values yet.


* :class:`pandas.Series` ``(data=None, index=None, dtype=None, name=None, copy=False, fastpath=False)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``data``
     - - Series type
       - List type
       - Array type
       - Constant Dictionary
       - None
     -
   * - ``index``
     - - SeriesType
     -
   * - ``dtype``
     - - Numpy or Pandas Type
       - String name for Numpy/Pandas Type
     - - **Must be constant at Compile Time**
       - String/Data Type must be one of the supported types (see ``Series.astype()``)
   * - ``name``
     - - String
     -

  .. note::

    If ``data`` is a Series and ``index`` is provided, implicit alignment is
    not performed yet.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f():
      ...     return pd.Series(np.arange(1000), dtype=np.float64, name="my_series")
      >>> f()
      0        0.0
      1        1.0
      2        2.0
      3        3.0
      4        4.0
            ...
      995    995.0
      996    996.0
      997    997.0
      998    998.0
      999    999.0
      Name: my_series, Length: 1000, dtype: float64


Attributes:
***********

* :attr:`pandas.Series.index`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.index
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      RangeIndex(start=0, stop=1000, step=1)

* :attr:`pandas.Series.values`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.values
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
              13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
              26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
              39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
              52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
              65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
              78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
              91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
             117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
             130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
             143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
             156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
             169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
             182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
             195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
             208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
             221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
             234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
             247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
             260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
             273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
             286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
             299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
             312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
             325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
             338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
             351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
             364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
             377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
             390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
             403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
             416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
             429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
             442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454,
             455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
             468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
             481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
             494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
             507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
             520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
             533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545,
             546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558,
             559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571,
             572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
             585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
             598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610,
             611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
             624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636,
             637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
             650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662,
             663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675,
             676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688,
             689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701,
             702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714,
             715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727,
             728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740,
             741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753,
             754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766,
             767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
             780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792,
             793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
             806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818,
             819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831,
             832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
             845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857,
             858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870,
             871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883,
             884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896,
             897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909,
             910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922,
             923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935,
             936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948,
             949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961,
             962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974,
             975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987,
             988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999])


* :attr:`pandas.Series.dtype` (object data types such as dtype of
  string series not supported yet)

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dtype
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      dtype('int64')

* :attr:`pandas.Series.shape`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.shape
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      (1000,)

* :attr:`pandas.Series.nbytes`

  .. note::
    This tracks the number of bytes used by Bodo which may differ
    from the Pandas values.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.nbytes
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      8000

* :attr:`pandas.Series.ndim`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.ndim
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      1

* :attr:`pandas.Series.size`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.size
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      1000

* :attr:`pandas.Series.T`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.T
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      0        0
      1        1
      2        2
      3        3
      4        4
            ...
      995    995
      996    996
      997    997
      998    998
      999    999
      Length: 1000, dtype: int64


* :meth:`pandas.Series.memory_usage` ``(index=True, deep=False)``

  `Supported Arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``index``
     - - Boolean
     - **Must be constant at Compile Time**

  .. note::
    This tracks the number of bytes used by Bodo which may differ
    from the Pandas values.


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.memory_usage()
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      8024

* :attr:`pandas.Series.hasnans`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.hasnans
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      False

* :attr:`pandas.Series.empty`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.empty
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      False

* :attr:`pandas.Series.dtypes`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dtypes
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      dtype('int64')

* :attr:`pandas.Series.name`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.name
      >>> S = pd.Series(np.arange(1000), name="my_series")
      >>> f(S)
      'my_series'

Conversion:
***********

* :meth:`pandas.Series.astype` ``(dtype, copy=True, errors="raise", _bodo_nan_to_str=True)``

  `Supported Arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``dtype``
     - - String (string must be parsable by ``np.dtype``)
       - Valid type (see types)
       - The following functions: float, int, bool, str
     - **Must be constant at Compile Time**
   * - ``copy``
     - - Boolean
     - **Must be constant at Compile Time**
   * - ``_bodo_nan_to_str``
     - - Boolean
     -  - **Must be constant at Compile Time**
        - Argument unique to Bodo. When ``True`` NA values in
          when converting to string are represented as NA instead
          of a string representation of the NA value (i.e. 'nan'),
          the default Pandas behavior.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.astype(np.float32)
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      0        0.0
      1        1.0
      2        2.0
      3        3.0
      4        4.0
            ...
      995    995.0
      996    996.0
      997    997.0
      998    998.0
      999    999.0
      Length: 1000, dtype: float32

* :meth:`pandas.Series.copy` ``(deep=True)``

  `Supported Arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``deep``
     - - Boolean

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.copy()
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      0        0
      1        1
      2        2
      3        3
      4        4
            ...
      995    995
      996    996
      997    997
      998    998
      999    999
      Length: 1000, dtype: int64

* :meth:`pandas.Series.to_numpy` ``(dtype=None, copy=False, na_value=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.to_numpy()
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
      array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
              13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
              26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
              39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
              52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
              65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
              78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
              91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
             104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
             117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
             130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
             143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
             156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
             169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
             182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
             195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
             208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
             221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
             234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
             247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
             260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
             273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
             286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
             299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
             312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
             325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337,
             338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350,
             351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
             364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376,
             377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389,
             390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402,
             403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415,
             416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
             429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441,
             442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454,
             455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467,
             468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
             481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493,
             494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506,
             507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519,
             520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532,
             533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545,
             546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558,
             559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571,
             572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584,
             585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597,
             598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610,
             611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623,
             624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636,
             637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
             650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662,
             663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675,
             676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688,
             689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701,
             702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714,
             715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727,
             728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740,
             741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753,
             754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766,
             767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
             780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792,
             793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805,
             806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818,
             819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831,
             832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843, 844,
             845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857,
             858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870,
             871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883,
             884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896,
             897, 898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 909,
             910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922,
             923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935,
             936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948,
             949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961,
             962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974,
             975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 985, 986, 987,
             988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999])

* :meth:`pandas.Series.to_list` ``()``

  .. note::
    Calling ``to_list`` on a non-float array with
    NA values with cause a runtime exception.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.to_list()
      >>> S = pd.Series(np.arange(50))
      >>> f(S)
      [0,
       1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9,
       10,
       11,
       12,
       13,
       14,
       15,
       16,
       17,
       18,
       19,
       20,
       21,
       22,
       23,
       24,
       25,
       26,
       27,
       28,
       29,
       30,
       31,
       32,
       33,
       34,
       35,
       36,
       37,
       38,
       39,
       40,
       41,
       42,
       43,
       44,
       45,
       46,
       47,
       48,
       49]



* :meth:`pandas.Series.tolist` ``()``

  .. note::
    Calling ``tolist`` on a non-float array with
    NA values with cause a runtime exception.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.tolist()
      >>> S = pd.Series(np.arange(50))
      >>> f(S)
      [0,
       1,
       2,
       3,
       4,
       5,
       6,
       7,
       8,
       9,
       10,
       11,
       12,
       13,
       14,
       15,
       16,
       17,
       18,
       19,
       20,
       21,
       22,
       23,
       24,
       25,
       26,
       27,
       28,
       29,
       30,
       31,
       32,
       33,
       34,
       35,
       36,
       37,
       38,
       39,
       40,
       41,
       42,
       43,
       44,
       45,
       46,
       47,
       48,
       49]


Indexing, iteration:
********************

Location based indexing using `[]`, `iat`, and `iloc` is supported.
Changing values of existing string Series using these operators
is not supported yet.

* :meth:`pandas.Series.iat`

We only support indexing using ``iat`` using a pair of integers

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, i):
      ...   return S.iat[i]
      >>> S = pd.Series(np.arange(1000))
      >>> f(S, 27)
      27

* :meth:`pandas.Series.iloc`

  `getitem`:

    - ``Series.iloc`` supports single integer indexing (returns a scalar) ``S.iloc[0]``

    - ``Series.iloc`` supports list/array/series of integers/bool (returns a Series) ``S.iloc[[0,1,2]]``

    - ``Series.iloc`` supports integer slice (returns a Series) ``S.iloc[[0:2]]``


  `setitem`:

    - Supports the same cases as getitem but the array type must be mutable (i.e. numeric array)

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, idx):
      ...   return S.iloc[idx]
      >>> S = pd.Series(np.arange(1000))
      >>> f(S, [1, 4, 29])
      1      1
      4      4
      29    29
      dtype: int64


* :meth:`pandas.Series.loc`

  `getitem`:

    - ``Series.loc`` supports list/array of booleans
    - ``Series.loc`` supports integer with RangeIndex

  `setitem`:

    - ``Series.loc`` supports list/array of booleans


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, idx):
      ...   return S.loc[idx]
      >>> S = pd.Series(np.arange(1000))
      >>> f(S, S < 10)
      0    0
      1    1
      2    2
      3    3
      4    4
      5    5
      6    6
      7    7
      8    8
      9    9
      dtype: int64

Binary operator functions:
**************************

* :meth:`pandas.Series.add` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.add`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.add(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1001
      1      1001
      2      1001
      3      1001
      4      1001
            ...
      995    1001
      996    1001
      997    1001
      998    1001
      999    1001
      Length: 1000, dtype: int64


* :meth:`pandas.Series.sub` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.sub`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.sub(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0     -999
      1     -997
      2     -995
      3     -993
      4     -991
            ...
      995    991
      996    993
      997    995
      998    997
      999    999
      Length: 1000, dtype: int64

* :meth:`pandas.Series.mul` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.mul`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.mul(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1000
      1      1998
      2      2994
      3      3988
      4      4980
            ...
      995    4980
      996    3988
      997    2994
      998    1998
      999    1000
      Length: 1000, dtype: int64

* :meth:`pandas.Series.div` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.div`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.div(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0         0.001000
      1         0.002002
      2         0.003006
      3         0.004012
      4         0.005020
                ...
      995     199.200000
      996     249.250000
      997     332.666667
      998     499.500000
      999    1000.000000
      Length: 1000, dtype: float64

* :meth:`pandas.Series.truediv` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.truediv`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.truediv(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0         0.001000
      1         0.002002
      2         0.003006
      3         0.004012
      4         0.005020
                ...
      995     199.200000
      996     249.250000
      997     332.666667
      998     499.500000
      999    1000.000000
      Length: 1000, dtype: float64

* :meth:`pandas.Series.floordiv` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.floordiv`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.floordiv(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0         0
      1         0
      2         0
      3         0
      4         0
            ...
      995     199
      996     249
      997     332
      998     499
      999    1000
      Length: 1000, dtype: int64

* :meth:`pandas.Series.mod` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.mod`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.mod(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1
      1      2
      2      3
      3      4
      4      5
            ..
      995    1
      996    1
      997    2
      998    1
      999    0
      Length: 1000, dtype: int64

* :meth:`pandas.Series.pow` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.pow`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.pow(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0                        1
      1                        0
      2     -5459658280481875879
      3                        0
      4      3767675092665006833
                    ...
      995        980159361278976
      996           988053892081
      997              994011992
      998                 998001
      999                   1000
      Length: 1000, dtype: int64

* :meth:`pandas.Series.radd` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.radd`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.radd(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1001
      1      1001
      2      1001
      3      1001
      4      1001
            ...
      995    1001
      996    1001
      997    1001
      998    1001
      999    1001
      Length: 1000, dtype: int64

* :meth:`pandas.Series.rsub` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rsub`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rsub(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      999
      1      997
      2      995
      3      993
      4      991
            ...
      995   -991
      996   -993
      997   -995
      998   -997
      999   -999
      Length: 1000, dtype: int64

* :meth:`pandas.Series.rmul` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rmul`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rmul(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1000
      1      1998
      2      2994
      3      3988
      4      4980
            ...
      995    4980
      996    3988
      997    2994
      998    1998
      999    1000
      Length: 1000, dtype: int64

* :meth:`pandas.Series.rdiv` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rdiv`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rdiv(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1000.000000
      1       499.500000
      2       332.666667
      3       249.250000
      4       199.200000
                ...
      995       0.005020
      996       0.004012
      997       0.003006
      998       0.002002
      999       0.001000
      Length: 1000, dtype: float64

* :meth:`pandas.Series.rtruediv` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rtruediv`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rtruediv(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1000.000000
      1       499.500000
      2       332.666667
      3       249.250000
      4       199.200000
                ...
      995       0.005020
      996       0.004012
      997       0.003006
      998       0.002002
      999       0.001000
      Length: 1000, dtype: float64

* :meth:`pandas.Series.rfloordiv` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rfloordiv`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rfloordiv(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1000
      1       499
      2       332
      3       249
      4       199
            ...
      995       0
      996       0
      997       0
      998       0
      999       0
      Length: 1000, dtype: int64

* :meth:`pandas.Series.rmod` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rmod`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rmod(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      0
      1      1
      2      2
      3      1
      4      1
            ..
      995    5
      996    4
      997    3
      998    2
      999    1
      Length: 1000, dtype: int64

* :meth:`pandas.Series.rpow` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.rpow`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.rpow(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0                     1000
      1                   998001
      2                994011992
      3             988053892081
      4          980159361278976
                    ...
      995    3767675092665006833
      996                      0
      997   -5459658280481875879
      998                      0
      999                      1
      Length: 1000, dtype: int64

* :meth:`pandas.Series.combine` ``(other, func, fill_value=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``other``
     - - Array
       - Series
     -
   * - ``func``
     - - Function that takes two scalar arguments and returns a scalar value.
     -
   * - ``fill_value``
     - - scalar
     - Must be provided if the Series lengths aren't equal and the dtypes aren't floats.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.combine(other, lambda a, b: 2 * a + b)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      1002
      1      1003
      2      1004
      3      1005
      4      1006
            ...
      995    1997
      996    1998
      997    1999
      998    2000
      999    2001
      Length: 1000, dtype: int64

* :meth:`pandas.Series.round` ``(decimals=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - Series with numeric data

  .. note::
    ``Series.round`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...   return S.round(2)
      >>> S = pd.Series(np.linspace(100, 1000))
      >>> f(S)
      0      100.00
      1      118.37
      2      136.73
      3      155.10
      4      173.47
      5      191.84
      6      210.20
      7      228.57
      8      246.94
      9      265.31
      10     283.67
      11     302.04
      12     320.41
      13     338.78
      14     357.14
      15     375.51
      16     393.88
      17     412.24
      18     430.61
      19     448.98
      20     467.35
      21     485.71
      22     504.08
      23     522.45
      24     540.82
      25     559.18
      26     577.55
      27     595.92
      28     614.29
      29     632.65
      30     651.02
      31     669.39
      32     687.76
      33     706.12
      34     724.49
      35     742.86
      36     761.22
      37     779.59
      38     797.96
      39     816.33
      40     834.69
      41     853.06
      42     871.43
      43     889.80
      44     908.16
      45     926.53
      46     944.90
      47     963.27
      48     981.63
      49    1000.00
      dtype: float64

* :meth:`pandas.Series.lt` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.lt`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.lt(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0       True
      1       True
      2       True
      3       True
      4       True
            ...
      995    False
      996    False
      997    False
      998    False
      999    False
      Length: 1000, dtype: bool

* :meth:`pandas.Series.gt` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.gt`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.gt(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      False
      1      False
      2      False
      3      False
      4      False
            ...
      995     True
      996     True
      997     True
      998     True
      999     True
      Length: 1000, dtype: bool

* :meth:`pandas.Series.le` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.le`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.le(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0       True
      1       True
      2       True
      3       True
      4       True
            ...
      995    False
      996    False
      997    False
      998    False
      999    False
      Length: 1000, dtype: bool

* :meth:`pandas.Series.ge` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.ge`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.ge(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      False
      1      False
      2      False
      3      False
      4      False
            ...
      995     True
      996     True
      997     True
      998     True
      999     True
      Length: 1000, dtype: bool

* :meth:`pandas.Series.ne` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.ne`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.ne(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      True
      1      True
      2      True
      3      True
      4      True
            ...
      995    True
      996    True
      997    True
      998    True
      999    True
      Length: 1000, dtype: bool

* :meth:`pandas.Series.eq` ``(other, level=None, fill_value=None, axis=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - numeric scalar
       - array with numeric data
       - Series with numeric data
   * - ``fill_value``
     - - numeric scalar

  .. note::
    ``Series.eq`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.eq(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      0      False
      1      False
      2      False
      3      False
      4      False
            ...
      995    False
      996    False
      997    False
      998    False
      999    False
      Length: 1000, dtype: bool

* :meth:`pandas.Series.dot` ``(other)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``other``
     - - Series with numeric data

  .. note::
    ``Series.dot`` is only supported on Series of numeric data.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...   return S.dot(other)
      >>> S = pd.Series(np.arange(1, 1001))
      >>> other = pd.Series(reversed(np.arange(1, 1001)))
      >>> f(S, other)
      167167000

Function application, GroupBy & Window:
***************************************

* :meth:`pandas.Series.apply` ``(func, convert_dtype=True, args=(), **kwargs)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``func``
     - - JIT function or callable defined within a JIT function
       - Numpy ufunc
       - Constant String which is the name of a supported Series method or Numpy ufunc
     - - Additional arguments for ``func`` can be passed as additional arguments.


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...   return S.apply(lambda x: x ** 0.75)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0.000000
      1      1.000000
      2      1.681793
      3      2.279507
      4      2.828427
              ...
      95    30.429352
      96    30.669269
      97    30.908562
      98    31.147239
      99    31.385308
      Length: 100, dtype: float64


* :meth:`pandas.Series.map` ``(arg, na_action=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``arg``
     - - Dictionary
       - JIT function or callable defined within a JIT function
       - Constant String which refers to a supported Series method or Numpy ufunc
       - Numpy ufunc

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...   return S.map(lambda x: x ** 0.75)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0.000000
      1      1.000000
      2      1.681793
      3      2.279507
      4      2.828427
              ...
      95    30.429352
      96    30.669269
      97    30.908562
      98    31.147239
      99    31.385308
      Length: 100, dtype: float64

* :meth:`pandas.Series.groupby` ``(by=None, axis=0, level=None, as_index=True, sort=True, group_keys=True, squeeze=NoDefault.no_default, observed=False, dropna=True)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``by``
      - - Array-like or Series data. This is not supported with Decimal or Categorical data.
      - - **Must be constant at Compile Time**
    * - ``level``
      - - integer
      - - **Must be constant at Compile Time**
        - Only ``level=0`` is supported and not with MultiIndex.


  .. important:

    You must provide exactly one of ``by`` and ``level``

  `Example Usage`:

    .. code:: ipython3

      >>> @bodo.jit
      ... def f(S, by_series):
      ...     return S.groupby(by_series).count()
      >>> S = pd.Series([1, 2, 24, None] * 5)
      >>> by_series = pd.Series(["421", "f31"] * 10)
      >>> f(S, by_series)

      421    10
      f31     5
      Name: , dtype: int64

  .. note::

    ``Series.groupby`` doesn't currently keep the name of the original Series.


* :meth:`pandas.Series.rolling` ``(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None, method='single')``

  (`window`, `min_periods` and `center` arguments supported)

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``window``
      - - Integer
        - String representing a Time Offset
        - Timedelta
    * - ``min_periods``
      - - Integer
    * - ``center``
      - - Boolean

  `Example Usage`:

  .. code:: ipython3

    >>> @bodo.jit
    ... def f(S):
    ...     return S.rolling(2).mean()
    >>> S = pd.Series(np.arange(100))
    >>> f(S)
    0      NaN
      1      0.5
      2      1.5
      3      2.5
      4      3.5
            ...
      95    94.5
      96    95.5
      97    96.5
      98    97.5
      99    98.5
      Length: 100, dtype: float64

* :meth:`pandas.Series.pipe` ``(func, *args, **kwargs)``

  `Supported arguments`:

  .. list-table::
      :widths: 25 35 40
      :header-rows: 1

      * - argument
        - datatypes
        - other requirements
      * - ``func``
        - - JIT function or callable defined within a JIT function.
        - - Additional arguments for ``func`` can be passed as additional arguments.


  .. note::

    ``func`` cannot be a tuple

  `Example Usage`:

    .. code:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     def g(row, y):
      ...         return row + y
      ...
      ...     def f(row):
      ...         return row * 2
      ...
      ...     return S.pipe(h).pipe(g, y=32)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      32
      1      34
      2      36
      3      38
      4      40
           ...
      95    222
      96    224
      97    226
      98    228
      99    230
      Length: 100, dtype: int64


Computations / Descriptive Stats:
*********************************

Statistical functions below are supported without optional arguments
unless support is explicitly mentioned.

* :meth:`pandas.Series.abs` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.abs()
      >>> S = (pd.Series(np.arange(100)) % 7) - 2
      >>> f(S)
      0     2
      1     1
      2     0
      3     1
      4     2
           ..
      95    2
      96    3
      97    4
      98    2
      99    1
      Length: 100, dtype: int64

* :meth:`pandas.Series.all` ``(axis=0, bool_only=None, skipna=True, level=None)``

  `Supported Arguments`: None

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.all()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      False

* :meth:`pandas.Series.any` ``(axis=0, bool_only=None, skipna=True, level=None)``

  `Supported Arguments`: None

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.any()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      True

* :meth:`pandas.Series.autocorr` ``(lag=1)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``lag``
      - - Integer

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.autocorr(3)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      -0.49872171657407155

* :meth:`pandas.Series.between` ``(left, right, inclusive='both')``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``left``
      - - Scalar matching the Series type
      -
    * - ``right``
      - - Scalar matching the Series type
      -
    * - ``inclusive``
      - - One of ("both", "neither")
      - - **Must be constant at Compile Time**

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.between(3, 5, "both")
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0     False
      1     False
      2     False
      3      True
      4      True
            ...
      95     True
      96     True
      97    False
      98    False
      99    False
      Length: 100, dtype: bool

* :meth:`pandas.Series.corr` ``(other, method='pearson', min_periods=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``other``
      - - Numeric Series or Array

  .. note::
    Series type must be numeric

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...     return S.cov(other)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> other = pd.Series(np.arange(100)) % 10
      >>> f(S, other)
      0.004326329627279103

* :meth:`pandas.Series.count` ``(level=None)``

  `Supported Arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.count()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      100

* :meth:`pandas.Series.cov` ``(other, min_periods=None, ddof=1)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``other``
      - - Numeric Series or Array
    * - ``ddof``
      - - Integer

  .. note::
    Series type must be numeric

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...     return S.cov(other)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> other = pd.Series(np.arange(100)) % 10
      >>> f(S, other)
      0.025252525252525252


* :meth:`pandas.Series.cummin` ``(axis=None, skipna=True)``

  `Supported Arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.cummin()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0     0
      1     0
      2     0
      3     0
      4     0
           ..
      95    0
      96    0
      97    0
      98    0
      99    0
      Length: 100, dtype: int64

* :meth:`pandas.Series.cummax` ``(axis=None, skipna=True)``

  `Supported Arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.cummax()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0     0
      1     1
      2     2
      3     3
      4     4
           ..
      95    6
      96    6
      97    6
      98    6
      99    6
      Length: 100, dtype: int64

* :meth:`pandas.Series.cumprod` ``(axis=None, skipna=True)``

  `Supported Arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.cumprod()
      >>> S = (pd.Series(np.arange(10)) % 7) + 1
      >>> f(S)
      0        1
      1        2
      2        6
      3       24
      4      120
      5      720
      6     5040
      7     5040
      8    10080
      9    30240
      dtype: int64

* :meth:`pandas.Series.cumsum` ``(axis=None, skipna=True)``

  `Supported Arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.cumsum()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0       0
      1       1
      2       3
      3       6
      4      10
           ...
      95    283
      96    288
      97    294
      98    294
      99    295
      Length: 100, dtype: int64

* :meth:`pandas.Series.describe` ``(percentiles=None, include=None, exclude=None, datetime_is_numeric=False)``

  `Supported Arguments`: None

  .. note::
    Bodo only supports numeric and datetime64 types and assumes  `datetime_is_numeric=True`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.describe()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      count    100.000000
      mean       2.950000
      std        2.021975
      min        0.000000
      25%        1.000000
      50%        3.000000
      75%        5.000000
      max        6.000000
      dtype: float64

* :meth:`pandas.Series.diff` ``(periods=1)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``periods``
      - - Integer

  .. note::
    Bodo only supports numeric and datetime64 types

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.diff(3)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0     NaN
      1     NaN
      2     NaN
      3     3.0
      4     3.0
           ...
      95    3.0
      96    3.0
      97    3.0
      98   -4.0
      99   -4.0
      Length: 100, dtype: float64

* :meth:`pandas.Series.kurt` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.kurt()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      -1.269562153611973

* :meth:`pandas.Series.mad` ``(axis=None, skipna=None, level=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.mad()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      1.736

* :meth:`pandas.Series.max` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.max()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      6

* :meth:`pandas.Series.mean` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.mean()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      2.95

* :meth:`pandas.Series.median` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.median()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      3.0

* :meth:`pandas.Series.min` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`: None

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.min()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0

* :meth:`pandas.Series.nlargest` ``(n=5, keep='first')``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``n``
      - - Integer

  .. note:: Series type must be numeric

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.nlargest(20)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      20    6
      27    6
      41    6
      34    6
      55    6
      13    6
      83    6
      90    6
      6     6
      69    6
      48    6
      76    6
      62    6
      97    6
      19    5
      5     5
      26    5
      61    5
      12    5
      68    5
      dtype: int64

* :meth:`pandas.Series.nsmallest` ``(n=5, keep='first')``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``n``
      - - Integer

  .. note::
    Series type must be numeric

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.nsmallest(20)
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      63    0
      7     0
      56    0
      98    0
      77    0
      91    0
      49    0
      42    0
      35    0
      84    0
      28    0
      21    0
      70    0
      0     0
      14    0
      43    1
      1     1
      57    1
      15    1
      36    1
      dtype: int64

* :meth:`pandas.Series.pct_change` ``(periods=1, fill_method='pad', limit=None, freq=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``periods``
      - - Integer

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to shift

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.pct_change(3)
      >>> S = (pd.Series(np.arange(100)) % 7) + 1
      >>> f(S)
      0          NaN
      1          NaN
      2          NaN
      3     3.000000
      4     1.500000
              ...
      95    1.500000
      96    1.000000
      97    0.750000
      98   -0.800000
      99   -0.666667
      Length: 100, dtype: float64

* :meth:`pandas.Series.prod` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.prod()
      >>> S = (pd.Series(np.arange(20)) % 3) + 1
      >>> f(S)
      93312

* :meth:`pandas.Series.product` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.product()
      >>> S = (pd.Series(np.arange(20)) % 3) + 1
      >>> f(S)
      93312

* :meth:`pandas.Series.quantile` ``(q=0.5, interpolation='linear')``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``q``
      - - Float in [0.0, 1.0]
        - Iterable of floats in [0.0, 1.0]

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.quantile([0.25, 0.5, 0.75])
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0.25    1.0
      0.50    3.0
      0.75    5.0
      dtype: float64

* :meth:`pandas.Series.sem` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean
    * - ``ddof``
      - - Integer

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.sem()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0.20219752318917852

* :meth:`pandas.Series.skew` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.skew()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0.032074996591991714

* :meth:`pandas.Series.std` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean
    * - ``ddof``
      - - Integer

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.std()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      2.021975231891785

* :meth:`pandas.Series.sum` ``(axis=None, skipna=None, level=None, numeric_only=None, min_count=0)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean
    * - ``min_count``
      - - Integer

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.sum()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      295

* :meth:`pandas.Series.var` ``(axis=None, skipna=None, level=None, ddof=1, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean
    * - ``ddof``
      - - Integer

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.var()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      4.088383838383838

* :meth:`pandas.Series.kurtosis` ``(axis=None, skipna=None, level=None, numeric_only=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``skipna``
      - - Boolean

  .. note::
    Series type must be numeric

  .. note::
    Bodo does not accept any additional arguments to pass to the function

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.kurtosis()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      -1.269562153611973

* :meth:`pandas.Series.unique` ``()``

  .. note::
    The output is assumed to be "small" relative to input and is replicated.
    Use ``Series.drop_duplicates()`` if the output should remain distributed.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.unique()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      [0 1 2 3 4 5 6]


* :meth:`pandas.Series.nunique` ``(dropna=True)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``dropna``
      - - Boolean

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.nunique()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      7

* :attr:`pandas.Series.is_monotonic`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.is_monotonic
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      True

* :attr:`pandas.Series.is_monotonic_increasing`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.is_monotonic_increasing
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      True

* :attr:`pandas.Series.is_monotonic_decreasing`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.is_monotonic_decreasing
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      False

* :meth:`pandas.Series.value_counts` ``(normalize=False, sort=True, ascending=False, bins=None, dropna=True)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``normalize``
      - - Boolean
      - - **Must be constant at Compile Time**
    * - ``sort``
      - - Boolean
      - - **Must be constant at Compile Time**
    * - ``ascending``
      - - Boolean
      -
    * - ``bins``
      - - Integer
        - Array-like of integers
      -

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.value_counts()
      >>> S = pd.Series(np.arange(100)) % 7
      >>> f(S)
      0    15
      1    15
      2    14
      3    14
      4    14
      5    14
      6    14
      dtype: int64



Reindexing / Selection / Label manipulation:
********************************************


* :meth:`pandas.Series.drop_duplicates` ``(keep='first', inplace=False)``

  `Supported arguments`: None

    `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.drop_duplicates()
      >>> S = pd.Series(np.arange(100)) % 10
      >>> f(S)
      0    0
      1    1
      2    2
      3    3
      4    4
      5    5
      6    6
      7    7
      8    8
      9    9
      dtype: int64


* :meth:`pandas.Series.equals` ``(other)``


  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``other``
      - - Series

  .. note::
    Series and ``other`` must contain scalar values in each row

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, other):
      ...     return S.equals(other)
      >>> S = pd.Series(np.arange(100)) % 10
      >>> other = pd.Series(np.arange(100)) % 5
      >>> f(S, other)
      False

* :meth:`pandas.Series.head` ``(n=5)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``n``
      - - Integer

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.head(10)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0    0
      1    1
      2    2
      3    3
      4    4
      5    5
      6    6
      7    7
      8    8
      9    9
      dtype: int64

* :meth:`pandas.Series.idxmax` ``(axis=0, skipna=True)``

  `Supported arguments`: None

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.idxmax()
      >>> S = pd.Series(np.arange(100))
      >>> S[(S % 3 == 0)] = 100
      >>> f(S)
      0

* :meth:`pandas.Series.idxmin` ``(axis=0, skipna=True)``

  `Supported arguments`: None

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.idxmin()
      >>> S = pd.Series(np.arange(100))
      >>> S[(S % 3 == 0)] = 100
      >>> f(S)
      1

* :meth:`pandas.Series.isin` ``(values)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``values``
      - - Series
        - Array
        - List

  .. note::
    `values` argument supports both distributed array/Series and replicated list/array/Series

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.isin([3, 11, 98])
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0     False
      1     False
      2     False
      3      True
      4     False
            ...
      95    False
      96    False
      97    False
      98     True
      99    False
      Length: 100, dtype: bool

* :meth:`pandas.Series.rename` ``(index=None, *, axis=None, copy=True, inplace=False, level=None, errors='ignore')``

    `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``index``
      - - String
    * - ``axis``
      - - Any value. Bodo ignores this argument entirely, which is consistent with Pandas.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.rename("a")
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0
      1      1
      2      2
      3      3
      4      4
            ..
      95    95
      96    96
      97    97
      98    98
      99    99
      Name: a, Length: 100, dtype: int64


* :meth:`pandas.Series.reset_index` ``(level=None, drop=False, name=None, inplace=False)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``level``
      - - Integer
        - Boolean
      - - **Must be constant at Compile Time**
    * - ``drop``
      - - Boolean
      - - **Must be constant at Compile Time**
        - If ``False``, Index name must be known at compilation time

  .. note::
    For MultiIndex case, only dropping all levels is supported.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.reset_index()
      >>> S = pd.Series(np.arange(100), index=pd.RangeIndex(100, 200, 1, name="b"))
      >>> f(S)
            b   0
      0   100   0
      1   101   1
      2   102   2
      3   103   3
      4   104   4
      ..  ...  ..
      95  195  95
      96  196  96
      97  197  97
      98  198  98
      99  199  99

      [100 rows x 2 columns]

* :meth:`pandas.Series.take` ``(indices, axis=0, is_copy=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``indices``
      - - Array like with integer data
      - - To have distributed data ``indices`` must be an array with the
          same distribution as S.

  .. note::
    Bodo does not accept any additional arguments for Numpy compatability

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.take([2, 7, 4, 19])
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      2      2
      7      7
      4      4
      19    19
      dtype: int64

* :meth:`pandas.Series.tail` ``(n=5)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``n``
      - - Integer

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.tail(10)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      90    90
      91    91
      92    92
      93    93
      94    94
      95    95
      96    96
      97    97
      98    98
      99    99
      dtype: int64

* :meth:`pandas.Series.where` ``(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``cond``
      - - boolean array
        - 1d bool numpy array
    * - ``other``
      - - 1d numpy array
        - scalar

  .. note::
    Series can contain categorical data if ``other`` is a scalar

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.where((S % 3) != 0, 0)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0
      1      1
      2      2
      3      0
      4      4
            ..
      95    95
      96     0
      97    97
      98    98
      99     0
      Length: 100, dtype: int64

* :meth:`pandas.Series.mask` ``(cond, other=nan, inplace=False, axis=None, level=None, errors='raise', try_cast=NoDefault.no_default)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``cond``
      - - boolean array
        - 1d bool numpy array
    * - ``other``
      - - 1d numpy array
        - scalar

  .. note::
    Series can contain categorical data if ``other`` is a scalar

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.mask((S % 3) != 0, 0)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0
      1      0
      2      0
      3      3
      4      0
            ..
      95     0
      96    96
      97     0
      98     0
      99    99
      Length: 100, dtype: int64

Missing data handling:
**********************

* :meth:`pandas.Series.backfill` ``(axis=None, inplace=False, limit=None, downcast=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.backfill()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0       1
      1       1
      2      -2
      3      -2
      4       5
      5       5
      6    <NA>
      dtype: Int64

* :meth:`pandas.Series.bfill` ``(axis=None, inplace=False, limit=None, downcast=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.bfill()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0       1
      1       1
      2      -2
      3      -2
      4       5
      5       5
      6    <NA>
      dtype: Int64

* :meth:`pandas.Series.dropna` ``(axis=0, inplace=False, how=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dropna()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      1     1
      3    -2
      5     5
      dtype: Int64

* :meth:`pandas.Series.ffill` ``(axis=None, inplace=False, limit=None, downcast=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.ffill()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0    <NA>
      1       1
      2       1
      3      -2
      4      -2
      5       5
      6       5
      dtype: Int64

* :meth:`pandas.Series.fillna` ``(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)``

  `Supported arguments`:

  .. list-table::
      :widths: 25 35 40
      :header-rows: 1

      * - argument
        - datatypes
        - other requirements
      * - ``value``
        - - Scalar
        -
      * - ``method``
        - - One of ("bfill", "backfill", "ffill", and "pad")
        - - **Must be constant at Compile Time**
      * - ``inplace``
        - - Boolean
        - - **Must be constant at Compile Time**

  .. note ::

    - If ``value`` is provided then ``method`` must be ``None`` and vice-versa
    - If ``method`` is provided then ``inplace`` must be ``False``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.fillna(-1)
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0    -1
      1     1
      2    -1
      3    -2
      4    -1
      5     5
      6    -1
      dtype: Int64


* :meth:`pandas.Series.isna` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.isna()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0     True
      1    False
      2     True
      3    False
      4     True
      5    False
      6     True
      dtype: bool

* :meth:`pandas.Series.isnull` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.isnull()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0     True
      1    False
      2     True
      3    False
      4     True
      5    False
      6     True
      dtype: bool


* :meth:`pandas.Series.notna` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.notna()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0    False
      1     True
      2    False
      3     True
      4    False
      5     True
      6    False
      dtype: bool

* :meth:`pandas.Series.notnull` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.notnull()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0    False
      1     True
      2    False
      3     True
      4    False
      5     True
      6    False
      dtype: bool

* :meth:`pandas.Series.pad` ``(axis=None, inplace=False, limit=None, downcast=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.pad()
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S)
      0    <NA>
      1       1
      2       1
      3      -2
      4      -2
      5       5
      6       5
      dtype: Int64

* :meth:`pandas.Series.replace` ``(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')``

  `Supported arguments`:

  .. list-table::
      :widths: 25 35 40
      :header-rows: 1

      * - argument
        - datatypes
        - other requirements
      * - ``to_replace``
        - - Scalar
          - List of Scalars
          - Dictionary mapping scalars of the same type
        -
      * - ``value``
        - - Scalar
        - If ``to_replace`` is not a scalar, value must be ``None``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S, replace_dict):
      ...     return S.replace(replace_dict)
      >>> S = pd.Series(pd.array([None, 1, None, -2, None, 5, None]))
      >>> f(S, {1: -2, -2: 5, 5: 27})
      0    <NA>
      1      -2
      2    <NA>
      3       5
      4    <NA>
      5      27
      6    <NA>
      dtype: Int64

Reshaping, sorting:
*******************

* :meth:`pandas.Series.argsort` ``(axis=0, kind='quicksort', order=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.sort_values()
      >>> S = pd.Series(np.arange(99, -1, -1), index=np.arange(100))
      >>> f(S)
      0     99
      1     98
      2     97
      3     96
      4     95
            ..
      95     4
      96     3
      97     2
      98     1
      99     0
      Length: 100, dtype: int64

* :meth:`pandas.Series.sort_values` ``(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False, key=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``ascending``
      - - Boolean
      -
    * - ``na_position``
      - - One of ("first", "last")
      - **Must be constant at Compile Time**

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.sort_values()
      >>> S = pd.Series(np.arange(99, -1, -1), index=np.arange(100))
      >>> f(S)
      99     0
      98     1
      97     2
      96     3
      95     4
            ..
      4     95
      3     96
      2     97
      1     98
      0     99
      Length: 100, dtype: int64


* :meth:`pandas.Series.sort_index` ``(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True, ignore_index=False, key=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``ascending``
      - - Boolean
      -
    * - ``na_position``
      - - One of ("first", "last")
      - **Must be constant at Compile Time**

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.sort_index()
      >>> S = pd.Series(np.arange(100), index=np.arange(99, -1, -1))
      >>> f(S)
      0     99
      1     98
      2     97
      3     96
      4     95
            ..
      95     4
      96     3
      97     2
      98     1
      99     0
      Length: 100, dtype: int64

* :meth:`pandas.Series.explode` ``(ignore_index=False)``

  `Supported arguments`: None

  .. note::
    Bodo's output type may differ from Pandas because Bodo
    must convert to a nullable type at compile time.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.explode()
      >>> S = pd.Series([np.arange(i) for i in range(10)])
      >>> f(S)
      0    <NA>
      1       0
      2       0
      2       1
      3       0
      3       1
      3       2
      4       0
      4       1
      4       2
      4       3
      5       0
      5       1
      5       2
      5       3
      5       4
      6       0
      6       1
      6       2
      6       3
      6       4
      6       5
      7       0
      7       1
      7       2
      7       3
      7       4
      7       5
      7       6
      8       0
      8       1
      8       2
      8       3
      8       4
      8       5
      8       6
      8       7
      9       0
      9       1
      9       2
      9       3
      9       4
      9       5
      9       6
      9       7
      9       8
      dtype: Int64

* :meth:`pandas.Series.repeat` ``(repeats, axis=None)``

  `Supported arguments`:

  .. list-table::
      :widths: 25 35
      :header-rows: 1

      * - argument
        - datatypes
      * - ``repeats``
        - - Integer
          - Array-like of integers the same length as the Series


  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.repeat(3)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      0
      0      0
      0      0
      1      1
      1      1
            ..
      98    98
      98    98
      99    99
      99    99
      99    99
      Length: 300, dtype: int64

Combining / comparing / joining / merging:
******************************************

* :meth:`pandas.Series.append` ``(to_append, ignore_index=False, verify_integrity=False)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - argument
      - datatypes
      - other requirements
    * - ``to_append``
      - - Series
        - List of Series
        - Tuple of Series
      -
    * - ``ignore_index``
      - - Boolean
      - **Must be constant at Compile Time**

  .. note ::
    Setting a name for the output Series is not supported yet

  .. important::
    Bodo currently concatenates local data chunks for distributed datasets, which does not preserve global order of concatenated objects in output.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S1, S2):
      ...     return S1.append(S2)
      >>> S = pd.Series(np.arange(100))
      >>> f(S, S)
      0      0
      1      1
      2      2
      3      3
      4      4
            ..
      95    95
      96    96
      97    97
      98    98
      99    99
      Length: 200, dtype: int64

Time series-related:

* :meth:`pandas.Series.shift` ``(periods=1, freq=None, axis=0, fill_value=None)``

  `Supported arguments`:

  .. list-table::
    :widths: 25 35
    :header-rows: 1

    * - argument
      - datatypes
    * - ``periods``
      - - Integer


  .. note::
    This data type for the series must be one of:
      - Integer
      - Float
      - Boolean
      - datetime.data
      - datetime64
      - timedelta64
      - string

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.shift(1)
      >>> S = pd.Series(np.arange(100))
      >>> f(S)
      0      NaN
      1      0.0
      2      1.0
      3      2.0
      4      3.0
            ...
      95    94.0
      96    95.0
      97    96.0
      98    97.0
      99    98.0
      Length: 100, dtype: float64




Datetime properties:
********************

* :attr:`pandas.Series.dt.date`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.date
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0     2022-01-01
      1     2022-01-01
      2     2022-01-01
      3     2022-01-01
      4     2022-01-02
      5     2022-01-02
      6     2022-01-02
      7     2022-01-03
      8     2022-01-03
      9     2022-01-03
      10    2022-01-04
      11    2022-01-04
      12    2022-01-04
      13    2022-01-05
      14    2022-01-05
      15    2022-01-05
      16    2022-01-05
      17    2022-01-06
      18    2022-01-06
      19    2022-01-06
      20    2022-01-07
      21    2022-01-07
      22    2022-01-07
      23    2022-01-08
      24    2022-01-08
      25    2022-01-08
      26    2022-01-09
      27    2022-01-09
      28    2022-01-09
      29    2022-01-10
      dtype: object

* :attr:`pandas.Series.dt.year`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.year
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     2022
      1     2022
      2     2022
      3     2022
      4     2022
      5     2022
      6     2022
      7     2022
      8     2022
      9     2022
      10    2023
      11    2023
      12    2023
      13    2023
      14    2023
      15    2023
      16    2023
      17    2023
      18    2023
      19    2023
      20    2024
      21    2024
      22    2024
      23    2024
      24    2024
      25    2024
      26    2024
      27    2024
      28    2024
      29    2025
      dtype: Int64


* :attr:`pandas.Series.dt.month`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.month
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0      1
      1      2
      2      3
      3      4
      4      6
      5      7
      6      8
      7      9
      8     11
      9     12
      10     1
      11     2
      12     4
      13     5
      14     6
      15     7
      16     9
      17    10
      18    11
      19    12
      20     2
      21     3
      22     4
      23     5
      24     7
      25     8
      26     9
      27    10
      28    12
      29     1
      dtype: Int64

* :attr:`pandas.Series.dt.day`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.day
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0      1
      1      8
      2     18
      3     25
      4      2
      5     10
      6     17
      7     24
      8      1
      9      9
      10    17
      11    24
      12     3
      13    11
      14    18
      15    26
      16     2
      17    10
      18    17
      19    25
      20     2
      21    11
      22    18
      23    26
      24     3
      25    10
      26    17
      27    25
      28     2
      29    10
      dtype: Int64

* :attr:`pandas.Series.dt.hour`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.hour
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0      0
      1      2
      2      4
      3      7
      4      9
      5     12
      6     14
      7     17
      8     19
      9     22
      10     0
      11     3
      12     5
      13     8
      14    10
      15    13
      16    15
      17    18
      18    20
      19    23
      20     1
      21     4
      22     6
      23     9
      24    11
      25    14
      26    16
      27    19
      28    21
      29     0
      dtype: Int64

* :attr:`pandas.Series.dt.minute`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.minute
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0      0
      1     28
      2     57
      3     26
      4     55
      5     24
      6     53
      7     22
      8     51
      9     20
      10    49
      11    18
      12    47
      13    16
      14    45
      15    14
      16    43
      17    12
      18    41
      19    10
      20    39
      21     8
      22    37
      23     6
      24    35
      25     4
      26    33
      27     2
      28    31
      29     0
      dtype: Int64

* :attr:`pandas.Series.dt.second`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.second
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0      0
      1     57
      2     55
      3     53
      4     51
      5     49
      6     47
      7     45
      8     43
      9     41
      10    39
      11    37
      12    35
      13    33
      14    31
      15    28
      16    26
      17    24
      18    22
      19    20
      20    18
      21    16
      22    14
      23    12
      24    10
      25     8
      26     6
      27     4
      28     2
      29     0
      dtype: Int64

* :attr:`pandas.Series.dt.microsecond`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.microsecond
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0          0
      1     931034
      2     862068
      3     793103
      4     724137
      5     655172
      6     586206
      7     517241
      8     448275
      9     379310
      10    310344
      11    241379
      12    172413
      13    103448
      14     34482
      15    965517
      16    896551
      17    827586
      18    758620
      19    689655
      20    620689
      21    551724
      22    482758
      23    413793
      24    344827
      25    275862
      26    206896
      27    137931
      28     68965
      29         0
      dtype: Int64

* :attr:`pandas.Series.dt.nanosecond`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.nanosecond
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0       0
      1     483
      2     966
      3     448
      4     932
      5     416
      6     896
      7     380
      8     864
      9     348
      10    832
      11    312
      12    792
      13    280
      14    760
      15    248
      16    728
      17    208
      18    696
      19    176
      20    664
      21    144
      22    624
      23    104
      24    584
      25     80
      26    560
      27     40
      28    520
      29      0
      dtype: Int64

* :attr:`pandas.Series.dt.week`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.week
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     52
      1      6
      2     11
      3     17
      4     22
      5     27
      6     33
      7     38
      8     44
      9     49
      10     3
      11     8
      12    14
      13    19
      14    24
      15    30
      16    35
      17    41
      18    46
      19    52
      20     5
      21    11
      22    16
      23    21
      24    27
      25    32
      26    38
      27    43
      28    49
      29     2
      dtype: Int64

* :attr:`pandas.Series.dt.weekofyear`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.weekofyear
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     52
      1      6
      2     11
      3     17
      4     22
      5     27
      6     33
      7     38
      8     44
      9     49
      10     3
      11     8
      12    14
      13    19
      14    24
      15    30
      16    35
      17    41
      18    46
      19    52
      20     5
      21    11
      22    16
      23    21
      24    27
      25    32
      26    38
      27    43
      28    49
      29     2
      dtype: Int64

* :attr:`pandas.Series.dt.day_of_week`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.day_of_week
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     5
      1     1
      2     4
      3     0
      4     3
      5     6
      6     2
      7     5
      8     1
      9     4
      10    1
      11    4
      12    0
      13    3
      14    6
      15    2
      16    5
      17    1
      18    4
      19    0
      20    4
      21    0
      22    3
      23    6
      24    2
      25    5
      26    1
      27    4
      28    0
      29    4
      dtype: Int64

* :attr:`pandas.Series.dt.weekday`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.weekday
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     5
      1     1
      2     4
      3     0
      4     3
      5     6
      6     2
      7     5
      8     1
      9     4
      10    1
      11    4
      12    0
      13    3
      14    6
      15    2
      16    5
      17    1
      18    4
      19    0
      20    4
      21    0
      22    3
      23    6
      24    2
      25    5
      26    1
      27    4
      28    0
      29    4
      dtype: Int64

* :attr:`pandas.Series.dt.dayofyear`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.dayofyear
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0       1
      1      39
      2      77
      3     115
      4     153
      5     191
      6     229
      7     267
      8     305
      9     343
      10     17
      11     55
      12     93
      13    131
      14    169
      15    207
      16    245
      17    283
      18    321
      19    359
      20     33
      21     71
      22    109
      23    147
      24    185
      25    223
      26    261
      27    299
      28    337
      29     10
      dtype: Int64

* :attr:`pandas.Series.dt.day_of_year`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.day_of_year
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0       1
      1      39
      2      77
      3     115
      4     153
      5     191
      6     229
      7     267
      8     305
      9     343
      10     17
      11     55
      12     93
      13    131
      14    169
      15    207
      16    245
      17    283
      18    321
      19    359
      20     33
      21     71
      22    109
      23    147
      24    185
      25    223
      26    261
      27    299
      28    337
      29     10
      dtype: Int64

* :attr:`pandas.Series.dt.quarter`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.quarter
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0     1
      1     1
      2     1
      3     2
      4     2
      5     3
      6     3
      7     3
      8     4
      9     4
      10    1
      11    1
      12    2
      13    2
      14    2
      15    3
      16    3
      17    4
      18    4
      19    4
      20    1
      21    1
      22    2
      23    2
      24    3
      25    3
      26    3
      27    4
      28    4
      29    1
      dtype: Int64

* :attr:`pandas.Series.dt.is_month_start`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_month_start
      >>> SS = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0      True
      1     False
      2     False
      3     False
      4      True
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25     True
      26    False
      27    False
      28    False
      29    False
      dtype: bool

* :attr:`pandas.Series.dt.is_month_end`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_month_end
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0     False
      1     False
      2     False
      3     False
      4     False
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25    False
      26    False
      27    False
      28    False
      29     True
      dtype: bool

* :attr:`pandas.Series.dt.is_quarter_start`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_quarter_start
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0      True
      1     False
      2     False
      3     False
      4     False
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25    False
      26    False
      27    False
      28    False
      29    False
      dtype: bool

* :attr:`pandas.Series.dt.is_quarter_end`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_quarter_end
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0     False
      1     False
      2     False
      3     False
      4     False
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25    False
      26    False
      27    False
      28    False
      29     True
      dtype: bool

* :attr:`pandas.Series.dt.is_year_start`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_year_start
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0      True
      1     False
      2     False
      3     False
      4     False
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25    False
      26    False
      27    False
      28    False
      29    False
      dtype: bool

* :attr:`pandas.Series.dt.is_year_end`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.is_year_end
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0     False
      1     False
      2     False
      3     False
      4     False
      5     False
      6     False
      7     False
      8     False
      9     False
      10    False
      11    False
      12    False
      13    False
      14    False
      15    False
      16    False
      17    False
      18    False
      19    False
      20    False
      21    False
      22    False
      23    False
      24    False
      25    False
      26    False
      27    False
      28    False
      29     True
      dtype: bool

* :attr:`pandas.Series.dt.daysinmonth`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.daysinmonth
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0     31
      1     28
      2     31
      3     30
      4     30
      5     31
      6     31
      7     30
      8     31
      9     31
      10    31
      11    28
      12    31
      13    31
      14    30
      15    31
      16    31
      17    31
      18    30
      19    31
      20    31
      21    31
      22    30
      23    31
      24    30
      25    31
      26    30
      27    31
      28    30
      29    31
      dtype: Int64

* :attr:`pandas.Series.dt.days_in_month`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.days_in_month
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='12/31/2024', periods=30))
      >>> f(S)
      0     31
      1     28
      2     31
      3     30
      4     30
      5     31
      6     31
      7     30
      8     31
      9     31
      10    31
      11    28
      12    31
      13    31
      14    30
      15    31
      16    31
      17    31
      18    30
      19    31
      20    31
      21    31
      22    30
      23    31
      24    30
      25    31
      26    30
      27    31
      28    30
      29    31
      dtype: Int64

Datetime methods:
*****************

* :meth:`pandas.Series.dt.normalize` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.normalize()
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0    2022-01-01
      1    2022-01-01
      2    2022-01-01
      3    2022-01-01
      4    2022-01-02
      5    2022-01-02
      6    2022-01-02
      7    2022-01-03
      8    2022-01-03
      9    2022-01-03
      10   2022-01-04
      11   2022-01-04
      12   2022-01-04
      13   2022-01-05
      14   2022-01-05
      15   2022-01-05
      16   2022-01-05
      17   2022-01-06
      18   2022-01-06
      19   2022-01-06
      20   2022-01-07
      21   2022-01-07
      22   2022-01-07
      23   2022-01-08
      24   2022-01-08
      25   2022-01-08
      26   2022-01-09
      27   2022-01-09
      28   2022-01-09
      29   2022-01-10
      dtype: datetime64[ns]

* :meth:`pandas.Series.dt.strftime` ``(date_format)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``date_format``
     - - String
     - Must be a valid `datetime format string <https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior>`_

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.strftime("%B %d, %Y, %r")
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0     January 01, 2022, 12:00:00 AM
      1     January 01, 2022, 07:26:53 AM
      2     January 01, 2022, 02:53:47 PM
      3     January 01, 2022, 10:20:41 PM
      4     January 02, 2022, 05:47:35 AM
      5     January 02, 2022, 01:14:28 PM
      6     January 02, 2022, 08:41:22 PM
      7     January 03, 2022, 04:08:16 AM
      8     January 03, 2022, 11:35:10 AM
      9     January 03, 2022, 07:02:04 PM
      10    January 04, 2022, 02:28:57 AM
      11    January 04, 2022, 09:55:51 AM
      12    January 04, 2022, 05:22:45 PM
      13    January 05, 2022, 12:49:39 AM
      14    January 05, 2022, 08:16:33 AM
      15    January 05, 2022, 03:43:26 PM
      16    January 05, 2022, 11:10:20 PM
      17    January 06, 2022, 06:37:14 AM
      18    January 06, 2022, 02:04:08 PM
      19    January 06, 2022, 09:31:02 PM
      20    January 07, 2022, 04:57:55 AM
      21    January 07, 2022, 12:24:49 PM
      22    January 07, 2022, 07:51:43 PM
      23    January 08, 2022, 03:18:37 AM
      24    January 08, 2022, 10:45:31 AM
      25    January 08, 2022, 06:12:24 PM
      26    January 09, 2022, 01:39:18 AM
      27    January 09, 2022, 09:06:12 AM
      28    January 09, 2022, 04:33:06 PM
      29    January 10, 2022, 12:00:00 AM
      dtype: object


* :meth:`pandas.Series.dt.round` ``(freq, ambiguous='raise', nonexistent='raise')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``freq``
     - - String
     - Must be a valid fixed `frequency alias <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.round("H")
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0    2022-01-01 00:00:00
      1    2022-01-01 07:00:00
      2    2022-01-01 15:00:00
      3    2022-01-01 22:00:00
      4    2022-01-02 06:00:00
      5    2022-01-02 13:00:00
      6    2022-01-02 21:00:00
      7    2022-01-03 04:00:00
      8    2022-01-03 12:00:00
      9    2022-01-03 19:00:00
      10   2022-01-04 02:00:00
      11   2022-01-04 10:00:00
      12   2022-01-04 17:00:00
      13   2022-01-05 01:00:00
      14   2022-01-05 08:00:00
      15   2022-01-05 16:00:00
      16   2022-01-05 23:00:00
      17   2022-01-06 07:00:00
      18   2022-01-06 14:00:00
      19   2022-01-06 22:00:00
      20   2022-01-07 05:00:00
      21   2022-01-07 12:00:00
      22   2022-01-07 20:00:00
      23   2022-01-08 03:00:00
      24   2022-01-08 11:00:00
      25   2022-01-08 18:00:00
      26   2022-01-09 02:00:00
      27   2022-01-09 09:00:00
      28   2022-01-09 17:00:00
      29   2022-01-10 00:00:00
      dtype: datetime64[ns]


* :meth:`pandas.Series.dt.floor` ``(freq, ambiguous='raise', nonexistent='raise')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``freq``
     - - String
     - Must be a valid fixed `frequency alias <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.floor("H")
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0    2022-01-01 00:00:00
      1    2022-01-01 07:00:00
      2    2022-01-01 14:00:00
      3    2022-01-01 22:00:00
      4    2022-01-02 05:00:00
      5    2022-01-02 13:00:00
      6    2022-01-02 20:00:00
      7    2022-01-03 04:00:00
      8    2022-01-03 11:00:00
      9    2022-01-03 19:00:00
      10   2022-01-04 02:00:00
      11   2022-01-04 09:00:00
      12   2022-01-04 17:00:00
      13   2022-01-05 00:00:00
      14   2022-01-05 08:00:00
      15   2022-01-05 15:00:00
      16   2022-01-05 23:00:00
      17   2022-01-06 06:00:00
      18   2022-01-06 14:00:00
      19   2022-01-06 21:00:00
      20   2022-01-07 04:00:00
      21   2022-01-07 12:00:00
      22   2022-01-07 19:00:00
      23   2022-01-08 03:00:00
      24   2022-01-08 10:00:00
      25   2022-01-08 18:00:00
      26   2022-01-09 01:00:00
      27   2022-01-09 09:00:00
      28   2022-01-09 16:00:00
      29   2022-01-10 00:00:00
      dtype: datetime64[ns]

* :meth:`pandas.Series.dt.ceil` ``(freq, ambiguous='raise', nonexistent='raise')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``freq``
     - - String
     - Must be a valid fixed `frequency alias <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.ceil("H")
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0    2022-01-01 00:00:00
      1    2022-01-01 08:00:00
      2    2022-01-01 15:00:00
      3    2022-01-01 23:00:00
      4    2022-01-02 06:00:00
      5    2022-01-02 14:00:00
      6    2022-01-02 21:00:00
      7    2022-01-03 05:00:00
      8    2022-01-03 12:00:00
      9    2022-01-03 20:00:00
      10   2022-01-04 03:00:00
      11   2022-01-04 10:00:00
      12   2022-01-04 18:00:00
      13   2022-01-05 01:00:00
      14   2022-01-05 09:00:00
      15   2022-01-05 16:00:00
      16   2022-01-06 00:00:00
      17   2022-01-06 07:00:00
      18   2022-01-06 15:00:00
      19   2022-01-06 22:00:00
      20   2022-01-07 05:00:00
      21   2022-01-07 13:00:00
      22   2022-01-07 20:00:00
      23   2022-01-08 04:00:00
      24   2022-01-08 11:00:00
      25   2022-01-08 19:00:00
      26   2022-01-09 02:00:00
      27   2022-01-09 10:00:00
      28   2022-01-09 17:00:00
      29   2022-01-10 00:00:00
      dtype: datetime64[ns]

* :meth:`pandas.Series.dt.month_name` ``(locale=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.month_name()
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2025', periods=30))
      >>> f(S)
      0       January
      1      February
      2         March
      3         April
      4          June
      5          July
      6        August
      7     September
      8      November
      9      December
      10      January
      11     February
      12        April
      13          May
      14         June
      15         July
      16    September
      17      October
      18     November
      19     December
      20     February
      21        March
      22        April
      23          May
      24         July
      25       August
      26    September
      27      October
      28     December
      29      January
      dtype: object

* :meth:`pandas.Series.dt.day_name` ``(locale=None)``

  `Supported arguments`: None

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.dt.day_name()
      >>> S = pd.Series(pd.date_range(start='1/1/2022', end='1/10/2022', periods=30))
      >>> f(S)
      0      Saturday
      1      Saturday
      2      Saturday
      3      Saturday
      4        Sunday
      5        Sunday
      6        Sunday
      7        Monday
      8        Monday
      9        Monday
      10      Tuesday
      11      Tuesday
      12      Tuesday
      13    Wednesday
      14    Wednesday
      15    Wednesday
      16    Wednesday
      17     Thursday
      18     Thursday
      19     Thursday
      20       Friday
      21       Friday
      22       Friday
      23     Saturday
      24     Saturday
      25     Saturday
      26       Sunday
      27       Sunday
      28       Sunday
      29       Monday
      dtype: object

String handling:
****************

* :meth:`pandas.Series.str.capitalize` ``()``

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.capitalize()
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1      Ce
      2     Erw
      3      A3
      4       @
      5     A n
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.center` ``(width, fillchar=' ')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``width``
     - - Integer
   * - ``fillchar``
     - - String with a single character

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.center(4)
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0     a
      1     ce
      2    Erw
      3     a3
      4     @
      5    a n
      6    ^ Ef
      dtype: object


* :meth:`pandas.Series.str.contains` ``(pat, case=True, flags=0, na=None, regex=True)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``pat``
     - - String
     -
   * - ``case``
     - - Boolean
     - **Must be constant at Compile Time**
   * - ``flags``
     - - Integer
     -
   * - ``regex``
     - - Boolean
     - **Must be constant at Compile Time**

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.contains("a.+")
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1    False
      2    False
      3     True
      4    False
      5     True
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.count` ``(pat, flags=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``pat``
     - - String
   * - ``flags``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.count("\w")
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    1
      1    2
      2    3
      3    2
      4    0
      5    2
      6    2
      dtype: Int64


* :meth:`pandas.Series.str.endswith` ``(pat, na=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``pat``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.endswith("e")
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1     True
      2    False
      3    False
      4    False
      5    False
      6    False
      dtype: boolean


* :meth:`pandas.Series.str.extract` ``(pat, flags=0, expand=True)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``pat``
     - - String
     - **Must be constant at Compile Time**
   * - ``flags``
     - - Integer
     - **Must be constant at Compile Time**
   * - ``expand``
     - - Boolean
     - **Must be constant at Compile Time**

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.extract("(a|e)")
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
           0
      0    a
      1    e
      2  NaN
      3    a
      4  NaN
      5    a
      6  NaN

* :meth:`pandas.Series.str.extractall` ``(pat, flags=0)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``pat``
     - - String
     - **Must be constant at Compile Time**
   * - ``flags``
     - - Integer
     - **Must be constant at Compile Time**

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.extractall("(a|n)")
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
               0
        match
      0 0      a
      3 0      a
      5 0      a
        1      n


* :meth:`pandas.Series.str.find` ``(sub, start=0, end=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``sub``
     - - String
   * - ``start``
     - - Integer
   * - ``end``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.find("a3", start=1)
      >>> S = pd.Series(["Aa3", "cea3", "14a3", " a3", "a3@", "a n3", "^ Ea3f"])
      >>> f(S)
      0     1
      1     2
      2     2
      3     1
      4    -1
      5    -1
      6     3
      dtype: Int64

* :meth:`pandas.Series.str.get` ``(i)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``i``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.get(1)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    NaN
      1      e
      2      4
      3    NaN
      4    NaN
      5
      6
      dtype: object

* :meth:`pandas.Series.str.join` ``(sep)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``sep``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.join(",")
      >>> S = pd.Series([["a", "fe", "@23"], ["a", "b"], [], ["c"]])
      >>> f(S)
      0    a,fe,@23
      1         a,b
      2
      3           c
      dtype: object

* :meth:`pandas.Series.str.len` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.len()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    1
      1    2
      2    2
      3    1
      4    1
      5    3
      6    4
      dtype: Int64

* :meth:`pandas.Series.str.ljust` ``(width, fillchar=' ')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``width``
     - - Integer
   * - ``fillchar``
     - - String with a single character

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.ljust(5, fillchar=",")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    A,,,,
      1    ce,,,
      2    14,,,
      3     ,,,,
      4    @,,,,
      5    a n,,
      6    ^ Ef,
      dtype: object

* :meth:`pandas.Series.str.lower` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.lower()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       a
      1      ce
      2      14
      3
      4       @
      5     a n
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.lstrip` ``(to_strip=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``to_strip``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.lstrip("c")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1       e
      2      14
      3
      4       @
      5     a n
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.pad` ``(width, side='left', fillchar=' ')``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``width``
     - - Integer
     -
   * - ``width``
     - - One of ("left", "right", "both")
     - **Must be constant at Compile Time**
   * - ``fillchar``
     - - String with a single character
     -

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.pad(5)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0        A
      1       ce
      2       14
      3
      4        @
      5      a n
      6     ^ Ef
      dtype: object

* :meth:`pandas.Series.str.repeat` ``(repeats)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``repeats``
     - - Integer
       - Array Like containing integers
     - If ``repeats`` is array like, then it must be the same length as the Series.

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.repeat(2)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0          AA
      1        cece
      2        1414
      3
      4          @@
      5      a na n
      6    ^ Ef^ Ef
      dtype: object

* :meth:`pandas.Series.str.replace` ``(pat, repl, n=- 1, case=None, flags=0, regex=None)``

  `regex` argument supported.

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.replace("(a|e)", "yellow")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0           A
      1     cyellow
      2          14
      3
      4           @
      5    yellow n
      6        ^ Ef
      dtype: object

* :meth:`pandas.Series.str.rfind` ``(sub, start=0, end=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``sub``
     - - String
   * - ``start``
     - - Integer
   * - ``end``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.rfind("a3", start=1)
      >>> S = pd.Series(["Aa3", "cea3", "14a3", " a3", "a3@", "a n3", "^ Ea3f"])
      >>> f(S)
      0     1
      1     2
      2     2
      3     1
      4    -1
      5    -1
      6     3
      dtype: Int64


* :meth:`pandas.Series.str.rjust` ``(width, fillchar=' ')``

  Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``width``
     - - Integer
   * - ``fillchar``
     - - String with a single character

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.rjust(10)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0             A
      1            ce
      2            14
      3
      4             @
      5           a n
      6          ^ Ef
      dtype: object

* :meth:`pandas.Series.str.rstrip` ``(to_strip=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``to_strip``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.rstrip("n")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1      ce
      2      14
      3
      4       @
      5      a
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.slice` ``(start=None, stop=None, step=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``start``
     - - Integer
   * - ``stop``
     - - Integer
   * - ``step``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.slice(1, 4)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    A
      1    c
      2    1
      3
      4    @
      5    a
      6    #
      dtype: object

* :meth:`pandas.Series.str.slice_replace` ``(start=None, stop=None, repl=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``start``
     - - Integer
   * - ``stop``
     - - Integer
   * - ``repl``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.slice_replace(1, 4)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    A
      1    c
      2    1
      3
      4    @
      5    a
      6    #
      dtype: object

* :meth:`pandas.Series.str.split` ``(pat=None, n=-1, expand=False)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``pat``
     - - String
   * - ``n``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.split(" ")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0        [A]
      1       [ce]
      2       [14]
      3       [, ]
      4        [@]
      5     [a, n]
      6    [#, Ef]
      dtype: object

* :meth:`pandas.Series.str.startswith` ``(pat, na=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``pat``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.startswith("A")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0     True
      1    False
      2    False
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.strip` ``(to_strip=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``to_strip``
     - - String

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.strip("n")
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1      ce
      2      14
      3
      4       @
      5      a
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.swapcase` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.swapcase()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       a
      1      CE
      2      14
      3
      4       @
      5     A N
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.title` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.title()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1      Ce
      2      14
      3
      4       @
      5     A N
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.upper` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.upper()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0       A
      1      CE
      2      14
      3
      4       @
      5     A N
      6    ^ Ef
      dtype: object

* :meth:`pandas.Series.str.zfill` ``(width)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35
   :header-rows: 1

   * - argument
     - datatypes
   * - ``width``
     - - Integer

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.zfill(5)
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    0000A
      1    000ce
      2    00014
      3    0000
      4    0000@
      5    00a n
      6    0^ Ef
      dtype: object

* :meth:`pandas.Series.str.isalnum` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isalnum()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0     True
      1     True
      2     True
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.isalpha` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isalpha()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0     True
      1     True
      2    False
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.isdigit` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isdigit()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1    False
      2     True
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.isspace` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isspace()
      >>> S = pd.Series(["A", "ce", "14", " ", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1    False
      2    False
      3     True
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.islower` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.islower()
      >>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1     True
      2    False
      3     True
      4    False
      5     True
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.isupper` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isupper()
      >>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0     True
      1    False
      2    False
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.istitle` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.istitle()
      >>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0     True
      1    False
      2    False
      3    False
      4    False
      5    False
      6     True
      dtype: boolean

* :meth:`pandas.Series.str.isnumeric` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isnumeric()
      >>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1    False
      2     True
      3    False
      4    False
      5    False
      6    False
      dtype: boolean

* :meth:`pandas.Series.str.isdecimal` ``()``

  .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.str.isdecimal()
      >>> S = pd.Series(["A", "ce", "14", "a3", "@", "a n", "^ Ef"])
      >>> f(S)
      0    False
      1    False
      2     True
      3    False
      4    False
      5    False
      6    False
      dtype: boolean


Categorical accessor:
*********************


* :attr:`pandas.Series.cat.codes`

  .. note::

    If categories cannot be determined at compile time, then Bodo defaults
    to creating codes with an ``int64``, which may differ from Pandas.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.cat.codes
      >>> S = pd.Series(["a", "ce", "Erw", "a3", "@"] * 10).astype("category")
      >>> f(S)
      0     2
      1     4
      2     1
      3     3
      4     0
      5     2
      6     4
      7     1
      8     3
      9     0
      10    2
      11    4
      12    1
      13    3
      14    0
      15    2
      16    4
      17    1
      18    3
      19    0
      20    2
      21    4
      22    1
      23    3
      24    0
      25    2
      26    4
      27    1
      28    3
      29    0
      30    2
      31    4
      32    1
      33    3
      34    0
      35    2
      36    4
      37    1
      38    3
      39    0
      40    2
      41    4
      42    1
      43    3
      44    0
      45    2
      46    4
      47    1
      48    3
      49    0
      dtype: int8

Serialization / IO / Conversion
*******************************

* :meth:`pandas.Series.to_csv` ``(path_or_buf=None, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', line_terminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)``
* :meth:`pandas.Series.to_dict` ``(into=<class 'dict'>)``

  `Supported arguments`: None

  .. note::
    - This method is not parallelized since dictionaries are not parallelized.
    - This method returns a typedDict, which maintains typing information if
      passing the dictionary between JIT code and regular Python. This can be
      converted to a regular Python dictionary by using the ``dict`` constructor.

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.to_dict()
      >>> S = pd.Series(np.arange(10))
      >>> dict(f(S))
      {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


* :meth:`pandas.Series.to_frame` ``(name=None)``

  `Supported arguments`:

  .. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - argument
     - datatypes
     - other requirements
   * - ``name``
     - - String
     - **Must be constant at Compile Time**

  .. note::
    If ``name`` is not provided Series name must be a known constant

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(S):
      ...     return S.to_frame("my_column")
      >>> S = pd.Series(np.arange(1000))
      >>> f(S)
           my_column
      0            0
      1            1
      2            2
      3            3
      4            4
      ..         ...
      995        995
      996        996
      997        997
      998        998
      999        999

[1000 rows x 1 columns]

.. _heterogeneous_series:

Heterogeneous Series
~~~~~~~~~~~~~~~~~~~~

Bodo's Series implementation requires all elements to share a common data type.
However, in situations where the size and types of the elements are constant at
compile time, Bodo has some mixed type handling with its Heterogeneous Series type.

.. warning::

  This type's primary purpose is for iterating through the rows of a DataFrame
  with different column types. You should not attempt to directly create Series
  with mixed types.

Heterogeneous Series operations are a subset of those supported for Series and
the supported operations are listed below. Please refer to :ref:`series` for
detailed usage.

Attributes:
***********

* :attr:`pandas.Series.index`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: len(row.index), axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     2
      1     2
      2     2
      3     2
      4     2
          ..
      95    2
      96    2
      97    2
      98    2
      99    2
      Length: 100, dtype: int64

* :attr:`pandas.Series.values`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.values, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0      (0, A)
      1      (1, b)
      2      (2, A)
      3      (3, b)
      4      (4, A)
            ...
      95    (95, b)
      96    (96, A)
      97    (97, b)
      98    (98, A)
      99    (99, b)
      Length: 100, dtype: object

* :attr:`pandas.Series.shape`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.shape, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     (2,)
      1     (2,)
      2     (2,)
      3     (2,)
      4     (2,)
            ...
      95    (2,)
      96    (2,)
      97    (2,)
      98    (2,)
      99    (2,)
      Length: 100, dtype: object

* :attr:`pandas.Series.ndim`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.ndim, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     1
      1     1
      2     1
      3     1
      4     1
          ..
      95    1
      96    1
      97    1
      98    1
      99    1
      Length: 100, dtype: int64

* :attr:`pandas.Series.size`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.size, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     2
      1     2
      2     2
      3     2
      4     2
          ..
      95    2
      96    2
      97    2
      98    2
      99    2
      Length: 100, dtype: int64

* :attr:`pandas.Series.T`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.T.size, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     2
      1     2
      2     2
      3     2
      4     2
          ..
      95    2
      96    2
      97    2
      98    2
      99    2
      Length: 100, dtype: int64

* :attr:`pandas.Series.empty`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.empty, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0     False
      1     False
      2     False
      3     False
      4     False
            ...
      95    False
      96    False
      97    False
      98    False
      99    False
      Length: 100, dtype: boolean

* :attr:`pandas.Series.name`

  `Example Usage`:

    .. code-block:: ipython3

      >>> @bodo.jit
      ... def f(df):
      ...     return df.apply(lambda row: row.name, axis=1)
      >>> df = pd.DataFrame({"A": np.arange(100), "B": ["A", "b"] * 50})
      >>> f(df)
      0      0
      1      1
      2      2
      3      3
      4      4
            ..
      95    95
      96    96
      97    97
      98    98
      99    99
      Length: 100, dtype: int64
