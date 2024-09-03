

number_128bit = 18897858802567400140233746507530646293
# number_128bit = 79228162532711081671548469249

hex_nr = "188a3315"


def print_binary(number, bits, print_number=True):
    binary = bin(number)[2:]
    binary = '0'*(bits-len(binary)) + binary
    binary_with_apostrophes = ''
    for i in range(len(binary)):
        binary_with_apostrophes += binary[i]
        if (i+1) % 32 == 0 and i != len(binary) - 1:
            binary_with_apostrophes += "'"
    if print_number:
        print("d",f"{number:40}",end=": ")
    print(binary_with_apostrophes)

def print_hex(number, bits, print_number=True):
    hex_str = hex(number)[2:]
    hex_str = '0'*(bits//4-len(hex_str)) + hex_str
    hex_with_apostrophes = ''
    for i in range(len(hex_str)):
        hex_with_apostrophes += hex_str[i]
        if (i+1) % 8 == 0 and i != len(hex_str) - 1:
            hex_with_apostrophes += "'"

    if print_number:
        print("d",f"{number:40}",end=": ")
    print(hex_with_apostrophes)

def print_hex_as_binary(hex_str):
    number = int(hex_str, 16)
    print("x",hex_str, end=": ")
    print_binary(number, 128, False)

print("128-bit number: initial number")
print_binary(number_128bit, 128)

print("hex number")
print_hex_as_binary(hex_nr)

print("----------")
print("128-bit number: initial number")
print_binary(number_128bit, 128)

print("hex number")
print_hex_as_binary("84321004")

print("---- >> 31 -----")
# print_binary(4294967296,128)
print_hex(4294967296,128)

print("----------")
# print_binary(8589934592,128)
print_hex(8589934592,128)

print("---- >> 30 ------")
# print_binary(17179869184,128)
print_hex(17179869184,128)
print("---- >> 29 ------")
# print_binary(34359738369,128)
print_hex(34359738369,128)
print("---- >> 28 ------")
# print_binary(79228162514264337662263427075,128)
print_hex(79228162514264337662263427075,128)
print("---- >> 27 ------")
# print_binary(237684487561239756996075323398,128)
print_hex(237684487561239756996075323398,128)
print("---- >> 26 ------")
# print_binary(554597137655190595659404148748,128)
print_hex(554597137655190595659404148748,128)


print("------ comparison ------")
print_hex_as_binary("000000001c6f30c00e30a80108642008")