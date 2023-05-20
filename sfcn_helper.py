def get_bin_range_step(age):
    """
    "The output layer contains 40 digits that represent the predicted probability that the subject’s age falls into
    a one-year age interval between 42 to 82 (for UK Biobank) or a two-year age interval between 14 to 94 (for PAC 2019)."


    :param age: age of the subject
    :return: the most suitable range of the bin, and the step size of the bin
    """
    UBB_margin = 2
    PAC_margin = 3
    if 42 + UBB_margin <= age <= 82 - UBB_margin:  # UK Biobank has default margin of 3,i.e., 44-80 real age
        return [42, 82], 1
    elif 14 + PAC_margin <= age <= 94 - PAC_margin:  # PAC 2019 has default margin of 3,i.e., 17-90 real age
        return [14, 94], 2
    else:
        print("Age out of range")
        return [14, 94], 2
