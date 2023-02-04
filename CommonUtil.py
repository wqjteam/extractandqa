
def get_first_index_in_array( be_search_array, target_array):
    a = be_search_array
    b = target_array
    index = 0

    while (index < len(a)):
        # 当两个数组形式对比的时候，有一个为ndraay 需要用到.all() 如果是两个纯array则不需要
        if a[index] == b[0] and (index + len(b)) <= len(a) and (a[index:index + len(b)] == b[:]):
            return index
        else:
            index += 1
    return -1