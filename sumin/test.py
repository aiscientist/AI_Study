def fizz_count(x):
    count = 0
    for item in x:
        if item == "fizz":
            count = count +1
    return count


test = ["fizz", "boy", "girl", "fizz", "fizz"]
print (fizz_count(test))
print ("hello")