import torch
import copy
import random

CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^',
            '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
            't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', u'口']

ind_to_chr = {k: v for k, v in enumerate(CTLABELS)}
chr_to_ind = {v: k for k, v in enumerate(CTLABELS)}


space = [' ']
separator = [',', '-', '_']
special = ['!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[',
           '\\', ']', '^', '`', '{', '|', '}', '~']
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', u'口']
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

query_space = torch.tensor([1 if c in space else 0 for c in CTLABELS] + [0])
query_separator = torch.tensor([1 if c in separator else 0 for c in CTLABELS] + [0])
query_special = torch.tensor([1 if c in special else 0 for c in CTLABELS] + [0])
query_alpha = torch.tensor([1 if c in alphabet else 0 for c in CTLABELS] + [0])
query_number = torch.tensor([1 if c in numbers else 0 for c in CTLABELS] + [0])
query_pad = torch.tensor([0 for _ in range(len(CTLABELS) + 1)])
query_pad[-1] = 1
query_empty = torch.tensor([0 for _ in range(len(CTLABELS) + 1)])


def compare_queries(query1, query2):
    return torch.all(torch.any(query1*query2, axis=1))


def max_query_types():
    return len(CTLABELS) + 1


def indices_to_text(indices):
    return "".join(ind_to_chr[index] if index != 96 else "" for index in indices)


def text_to_indices(text, pad=25):
    return [chr_to_ind[c] for c in text] + [96 for _ in range(pad - len(text))]


def text_to_query_t1(text, indices=True):
    """
    First type of query, should function the same way as in the previous version. We put a 1 in all the characters of
    the same type (space, special, chars and number)
    :param text: list of indices/text
    :param indices: True if it's indices instead of characters
    :return: the multi-hot encoded query
    """
    query = []

    for index in text:
        if index == 96 and indices:
            query.append(copy.deepcopy(query_pad))
            continue

        if indices:
            if index in ind_to_chr.keys():
                char = ind_to_chr[index]
            else:
                print("wrong value", index, char)
                raise
        else:
            char = index

        if char in space:
            query.append(copy.deepcopy(query_space))
        elif char in separator:
            query.append(copy.deepcopy(query_separator))
        elif char in special:
            query.append(copy.deepcopy(query_special))
        elif char in alphabet:
            query.append(copy.deepcopy(query_alpha))
        elif char in numbers:
            query.append(copy.deepcopy(query_number))

    return torch.stack(query)


def generate_query(types, mask_force):
    query = []

    assert(len(types) == len(mask_force))

    for type, force in zip(types, mask_force):
        if force:
            current = copy.deepcopy(query_empty)
            index = chr_to_ind[type]
            current[index] = 1

        else:
            if type == "s":  # space
                current = copy.deepcopy(query_space)
            elif type == "e":  # separator
                current = copy.deepcopy(query_separator)
            elif type == "p":  # special
                current = copy.deepcopy(query_separator)
            elif type == "l":  # letter
                current = copy.deepcopy(query_alpha)
            elif type == "n":  # number
                current = copy.deepcopy(query_number)
            else:
                print("incorrect query type")
                exit(0)

        query.append(current)
    query += [query_pad for _ in range(25 - len(query))]

    return torch.stack(query)


def random_char_query(p, max_chars=max_query_types()):
    """
    Returns a random character query.
    :param p: for each character we calculate a random value between 0 and 1, if this
    value > p we set that character to 0, otherwise 1. If, for example, p=0.25, the probability of setting a character
    to 1 is 25%.
    :param max_chars: number of characters, should just be len(CTLABELS) + pad_char.
    :return: random character query.
    """
    return torch.tensor([0 if p < random.random() else 1 for _ in range(max_chars)])


def text_to_query_t2(text, mask, indices=True, include_others=0.0):
    """
    Second type of query. Following the positions of the mask, sets all the query characters of the selected positions to
    0 except for the GT character, which is set to 1. This should force the model to find words that exactly match the
    characters that have been set to 1. For example, if the character at position n of the GT transcription is "A", the
    query returned by text_to_query_t1 would be:

    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

    where the alphabet characters are all set to 1. Instead, if the position n of the mask is set to 1, this function
    will return:
                                                                                             Only A set to 1   ↓
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    forcing the character at position n to be the letter A. We can force the model, for example, to find words that end
    with "KG", start with "000", etc.

    To make things more fun, you can also randomly include other characters of the same type. include_others controls
    the probability of adding characters of the same type. For example, if in the previous example we set
    include_others=0.5, there's a 0.5 probability of adding characters of the same type. A possible output would be:

                                                                                      This is still set to 1   ↓
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
            1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0])

    The indices that have been randomly set to 1 are all alphabet characters, given that the letter A is of this type.
    During inference, this would allow to target multiple characters at the same time, for example words ending with
    "kg" or "KG", or that start with "000" or "111", or start with "GB" or "ES" AND end with "4" or "5"...

    NOTE: IF THE MASK IS TO ALL 1s, THIS FUNCTION RETURNS THE 1-HOT ENCODING OF THE TRANSCRIPTION (USEFUL TO COMPARE
    queries AND TEXT USING compare_queries).

    :param text: list of indices/text
    :param mask: positions set to 1 or True will force the character query to be equal to the ground truth transcription.
    The length of the mask must be equal to the text.
    :param indices: True if it's indices instead of characters
    :param include_others: probability of randomly adding characters of the same type
    :return: the multi-hot encoded query
    """
    assert(len(text) == len(mask))

    query = []
    for index, force_gt in zip(text, mask):
        if index == 96 and indices:  # for pad indices we don't look at the mask
            query.append(copy.deepcopy(query_pad))
            continue

        if indices:
            if index in ind_to_chr.keys():
                char = ind_to_chr[index]
            else:
                print("wrong value", index, char)
                raise
        else:
            char = index

        if force_gt:
            current = copy.deepcopy(query_empty)
            current[index] = 1

            if include_others > 0.0:
                rand_query = random_char_query(include_others)
                if char in space:
                    additional_characters = query_space
                if char in separator:
                    additional_characters = copy.deepcopy(query_separator) * rand_query
                elif char in special:
                    additional_characters = copy.deepcopy(query_special) * rand_query
                elif char in alphabet:
                    additional_characters = copy.deepcopy(query_alpha) * rand_query
                elif char in numbers:
                    additional_characters = copy.deepcopy(query_number) * rand_query

                current = torch.clamp(current + additional_characters, max=1.0)
                assert(current[index] == 1.0)
                assert(torch.max(current).item() == 1.0)

            query.append(current)

        else:
            if char in space:
                query.append(copy.deepcopy(query_space))
            elif char in separator:
                query.append(copy.deepcopy(query_separator))
            elif char in special:
                query.append(copy.deepcopy(query_special))
            elif char in alphabet:
                query.append(copy.deepcopy(query_alpha))
            elif char in numbers:
                query.append(copy.deepcopy(query_number))

    return torch.stack(query)


extra_types = {query_space, query_separator, query_special, query_number}
def text_to_query_t3(text, mask, indices=True):
    """
    Third type of query. Following the elements of the mask set to 1, sets the characters of another type to 1 (jointly
    with the characters of the class selected). If the mask is 0, the query produced for that character follows is the
    same approach as in text_to_query_t1. For example, if our character is a letter and the mask element is 0, the output
    is:

    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

    If the mask position for that character is set to 1, this function sets to 1 the characters of another type at
    random. For example, since our example case is a character, the function will randomly set to 1 the characters of
    one of these classes: [query_space, query_separator, query_special, query_number]. If the number class is selected, the
    resulting query would be:

    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0])

    Where the model has selected letters and numbers. During inference, this should make the model accept different
    types of characters at certain positions.

    :param text: list of indices/text
    :param mask: positions set to 1 or True will force the model to add another type of character in that position. The
    type is selected at random.
    :param indices: True if it's indices instead of characters
    :return: the multi-hot encoded query
    """
    assert(len(text) == len(mask))

    query = []
    for index, include_other in zip(text, mask):
        if index == 96 and indices:
            query.append(copy.deepcopy(query_pad))
            continue

        if indices:
            if index in ind_to_chr.keys():
                char = ind_to_chr[index]
            else:
                print("wrong value", index, char)
                raise
        else:
            char = index

        if char in space:
            current_query = copy.deepcopy(query_space)
            if include_other:
                current_query += random.choice(list(extra_types - {query_space}))
        elif char in separator:
            current_query = copy.deepcopy(query_separator)
            if include_other:
                current_query += random.choice(list(extra_types - {query_separator}))
        elif char in special:
            current_query = copy.deepcopy(query_special)
            if include_other:
                current_query += random.choice(list(extra_types - {query_special}))
        elif char in alphabet:
            current_query = copy.deepcopy(query_alpha)
            if include_other:
                current_query += random.choice(list(extra_types))
        elif char in numbers:
            current_query = copy.deepcopy(query_number)
            if include_other:
                current_query += random.choice(list(extra_types - {query_number}))
        query.append(current_query)

    return torch.stack(query)


if __name__ == "__main__":
    indices1 = text_to_indices("12345")
    indices2 = text_to_indices("123ab")
    query1 = text_to_query_t1(indices1)
    query2 = text_to_query_t1(indices2)
    print(compare_queries(query1, query2).item())  # False

    indices1 = text_to_indices("1234b")
    indices2 = text_to_indices("1235a")
    query1 = text_to_query_t1(indices1)
    query2 = text_to_query_t1(indices2)
    print(compare_queries(query1, query2).item())  # True

    indices1 = text_to_indices("ab12 4_5#")
    indices2 = text_to_indices("cd12 5-6!")
    query1 = text_to_query_t1(indices1)
    query2 = text_to_query_t1(indices2)
    print(compare_queries(query1, query2).item())  # True

    indices1 = text_to_indices("HelloWorld")
    indices2 = text_to_indices("JelloWorld")
    indices3 = text_to_indices("Hehehehehe")
    mask = [0 for _ in range(len(indices2))]
    mask[0] = 1
    query1 = text_to_query_t2(indices1, mask=mask)
    query2 = text_to_query_t2(indices2, mask=mask)
    query3 = text_to_query_t2(indices3, mask=mask)
    print(compare_queries(query1, query2).item(), compare_queries(query1, query3).item())  # False, True

    indices1 = text_to_indices("HelloWorld")
    indices2 = text_to_indices("JelloWorld")
    indices3 = text_to_indices("HleolDlorw")
    mask = [0 for _ in range(len(indices2))]
    mask[0] = 1
    query1 = text_to_query_t2(indices1, mask=mask)
    # Compare with one-hot encodings
    query2 = text_to_query_t2(indices2, mask=[1 for _ in range(len(indices1))])
    query3 = text_to_query_t2(indices3, mask=[1 for _ in range(len(indices1))])
    print(compare_queries(query1, query2).item(), compare_queries(query1, query3).item())  # False, True

    query1 = text_to_query_t2(indices1, mask=mask, include_others=0.5)
    # the query first position should have the character index of H to 1, the comparison below should return True
    print(query1[0], (query1[0][indices1[0]] == 1).item())  # True
    print(compare_queries(query1, query3).item())  # True

    indices1 = text_to_indices("HelloWorld")
    indices2 = text_to_indices("1elloWorld")
    indices3 = text_to_indices("-elloWorld")
    indices4 = text_to_indices(" elloWorld")
    indices5 = text_to_indices("+elloWorld")
    mask = [0 for _ in range(len(indices2))]
    mask[0] = 1
    query1 = text_to_query_t3(indices1, mask=mask)
    query2 = text_to_query_t1(indices2)
    query3 = text_to_query_t1(indices3)
    query4 = text_to_query_t1(indices4)
    query5 = text_to_query_t1(indices5)
    print(compare_queries(query1, query2).item(), compare_queries(query1, query3).item(),
          compare_queries(query1, query4).item(), compare_queries(query1, query5).item())  # one should be True

    exit(0)
