def performance(y_test, y_pre):
    sum = 0.0
    j = 0
    for i in y_test:
        sum = sum + 1 - abs(i-y_pre[j])/i
        j = j + 1
    sum = sum*100/len(y_test)
    return sum
