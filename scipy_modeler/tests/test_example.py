# an example

# 3 steps setup -> excercise -> verify
# clean up with pytest is not required.

def test_title_case():

    # setup
    in_string = 'this is a test string'
    desired = 'This Is A Test String'

    # excercise
    actual = in_string.title()

    # verify
    assert actual == desired