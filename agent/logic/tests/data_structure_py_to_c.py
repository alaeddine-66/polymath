# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from agent.logic.logic_py_c_data_structure_generator import (
    LogicPyCDataStructureGenerator,
)
from libcst import Module, parse_module


class TestLogicPyCDataStructureGenerator(TestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_6x6(self) -> None:
        self.__test_harness(
            """
class House:
    number: Unique[Domain[int, range(1, 3)]]          # houses 1 and 2
    name: Unique[Domain[str, "Alice", "Eric"]]        # one of each name
    occupation: Unique[Domain[str, "Teacher", "Doctor"]]
    genre: Unique[Domain[str, "Fantasy", "Mystery"]]  # each genre used once

class Solution:
    houses: list[House, 2]  
    
def validate(solution: Solution) -> None:
    # The teacher lives directly to the left of the fantasy-book lover
    teacher_house = nondet(solution.houses)
    assume(teacher_house.occupation == "Teacher")

    fantasy_house = nondet(solution.houses)
    assume(fantasy_house.genre == "Fantasy")

    # "Directly to the left" (smaller house number by exactly 1)
    assert teacher_house.number + 1 == fantasy_house.number

    # Eric is the teacher
    eric_house = nondet(solution.houses)
    assume(eric_house.name == "Eric")
    assert eric_house.occupation == "Teacher"

    # Alice loves mystery books
    alice_house = nondet(solution.houses)
    assume(alice_house.name == "Alice")
    assert alice_house.genre == "Mystery"
""",
            """struct House {
    int house_number;
    const char * name;
    const char * smoothie;
    const char * lunch;
    const char * phone;
    const char * car;
    const char * house_style;
};

static int House_house_number[] = {1, 2, 3, 4, 5, 6};
static bool House_house_number_used[6];
static const char * House_name[] = {"Alice", "Eric", "Peter", "Carol", "Bob", "Arnold"};
static bool House_name_used[6];
static const char * House_smoothie[] = {"watermelon", "blueberry", "desert", "cherry", "dragonfruit", "lime"};
static bool House_smoothie_used[6];
static const char * House_lunch[] = {"stew", "pizza", "grilled cheese", "stir fry", "soup", "spaghetti"};
static bool House_lunch_used[6];
static const char * House_phone[] = {"google pixel 6", "iphone 13", "xiaomi mi 11", "huawei p50", "samsung galaxy s21", "oneplus 9"};
static bool House_phone_used[6];
static const char * House_car[] = {"tesla model 3", "honda civic", "toyota camry", "ford f150", "chevrolet silverado", "bmw 3 series"};
static bool House_car_used[6];
static const char * House_house_style[] = {"craftsman", "ranch", "modern", "victorian", "mediterranean", "colonial"};
static bool House_house_style_used[6];

static void init_House(struct House * instance) {
    __CPROVER_unique_domain(instance->house_number, House_house_number);
    __CPROVER_unique_domain(instance->name, House_name);
    __CPROVER_unique_domain(instance->smoothie, House_smoothie);
    __CPROVER_unique_domain(instance->lunch, House_lunch);
    __CPROVER_unique_domain(instance->phone, House_phone);
    __CPROVER_unique_domain(instance->car, House_car);
    __CPROVER_unique_domain(instance->house_style, House_house_style);
}

struct PuzzleSolution {
    struct House houses[6];
};

static void init_PuzzleSolution(struct PuzzleSolution * instance) {
    for (size_t i = 0; i < sizeof(instance->houses) / sizeof(instance->houses[0]); ++i) {
        init_House(&instance->houses[i]);
    }
}

""",
        )

    def test_primitive_list(self) -> None:
        self.__test_harness(
            """class House:
    number: Unique[Domain[int, range(1, 7)]]  # House numbers from 1 to 6
    name: Unique[Domain[str, "Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"]]
    music_genre: Unique[Domain[str, "classical", "hip hop", "jazz", "pop", "rock", "country"]]
    mother: Unique[Domain[str, "Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"]]
    child: Unique[Domain[str, "Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"]]
    height: Unique[Domain[str, "very short", "tall", "short", "very tall", "super tall", "average"]]
    animal: Unique[Domain[str, "bird", "dog", "horse", "rabbit", "cat", "fish"]]

class Solution:
    header: list[str, 7] = ["House", "Name", "MusicGenre", "Mother", "Child", "Height", "Animal"]
    rows: list[House, 6]
""",
            """struct House {
    int number;
    const char * name;
    const char * music_genre;
    const char * mother;
    const char * child;
    const char * height;
    const char * animal;
};

static int House_number[] = {1, 2, 3, 4, 5, 6};
static bool House_number_used[6];
static const char * House_name[] = {"Alice", "Carol", "Eric", "Peter", "Bob", "Arnold"};
static bool House_name_used[6];
static const char * House_music_genre[] = {"classical", "hip hop", "jazz", "pop", "rock", "country"};
static bool House_music_genre_used[6];
static const char * House_mother[] = {"Sarah", "Penny", "Aniya", "Janelle", "Kailyn", "Holly"};
static bool House_mother_used[6];
static const char * House_child[] = {"Alice", "Fred", "Timothy", "Bella", "Samantha", "Meredith"};
static bool House_child_used[6];
static const char * House_height[] = {"very short", "tall", "short", "very tall", "super tall", "average"};
static bool House_height_used[6];
static const char * House_animal[] = {"bird", "dog", "horse", "rabbit", "cat", "fish"};
static bool House_animal_used[6];

static void init_House(struct House * instance) {
    __CPROVER_unique_domain(instance->number, House_number);
    __CPROVER_unique_domain(instance->name, House_name);
    __CPROVER_unique_domain(instance->music_genre, House_music_genre);
    __CPROVER_unique_domain(instance->mother, House_mother);
    __CPROVER_unique_domain(instance->child, House_child);
    __CPROVER_unique_domain(instance->height, House_height);
    __CPROVER_unique_domain(instance->animal, House_animal);
}

struct Solution {
    const char * header[7];
    struct House rows[6];
};

static void init_Solution(struct Solution * instance) {
    __CPROVER_array_copy(instance->header, (const char *[]){"House", "Name", "MusicGenre", "Mother", "Child", "Height", "Animal"});
    for (size_t i = 0; i < sizeof(instance->rows) / sizeof(instance->rows[0]); ++i) {
        init_House(&instance->rows[i]);
    }
}

""",
        )

    def test_scalar_fields(self) -> None:
        self.__test_harness(
            """class Person:
    name: Unique[Domain[str, "Alice", "Eric", "Arnold", "Peter"]]
    occupation: Unique[Domain[str, "artist", "engineer", "teacher", "doctor"]]
    book_genre: Unique[Domain[str, "fantasy", "science fiction", "mystery", "romance"]]
    phone_model: Unique[Domain[str, "google pixel 6", "iphone 13", "oneplus 9", "samsung galaxy s21"]]
    age: int = 10

class House:
    id: Unique[Domain[int, range(1, 5)]]
    person: Person

class Solution:
    houses: list[House, 4]

def validate(solution: Solution) -> None:
    # Clue 1: The person who is an engineer is directly left of the person who uses a Samsung Galaxy S21.
    engineer = nondet(solution.houses)
    assume(engineer.person.occupation == "engineer")
    samsung_user = nondet(solution.houses)
    assume(samsung_user.person.phone_model == "samsung galaxy s21")
    assert engineer.id + 1 == samsung_user.id

    # Clue 2: The person who loves fantasy books is in the second house.
    fantasy_lover = nondet(solution.houses)
    assume(fantasy_lover.person.book_genre == "fantasy")
    assert fantasy_lover.id == 2

    # Clue 3: Alice is not in the second house.
    alice = nondet(solution.houses)
    assume(alice.person.name == "Alice")
    assert alice.id != 2

    # Clue 4: Eric is the person who is a teacher.
    eric = nondet(solution.houses)
    assume(eric.person.name == "Eric")
    assert eric.person.occupation == "teacher"

    # Clue 5: The person who uses a Samsung Galaxy S21 is the person who loves fantasy books.
    samsung_user = nondet(solution.houses)
    assume(samsung_user.person.phone_model == "samsung galaxy s21")
    assert samsung_user.person.book_genre == "fantasy"

    # Clue 6: The person who uses an iPhone 13 is the person who loves science fiction books.
    iphone_user = nondet(solution.houses)
    assume(iphone_user.person.phone_model == "iphone 13")
    assert iphone_user.person.book_genre == "science fiction"

    # Clue 7: The person who loves science fiction books is somewhere to the left of the person who uses a OnePlus 9.
    science_fiction_lover = nondet(solution.houses)
    assume(science_fiction_lover.person.book_genre == "science fiction")
    oneplus_user = nondet(solution.houses)
    assume(oneplus_user.person.phone_model == "oneplus 9")
    assert science_fiction_lover.id < oneplus_user.id

    # Clue 8: The person who uses a OnePlus 9 is Arnold.
    oneplus_user = nondet(solution.houses)

    assume(oneplus_user.person.phone_model == "oneplus 9")
    assert oneplus_user.person.name == "Arnold"

    # Clue 9: The person who is a doctor is the person who loves mystery books.
    doctor = nondet(solution.houses)
    assume(doctor.person.occupation == "doctor")
    assert doctor.person.book_genre == "mystery"

    # Clue 10: The person who uses an iPhone 13 is the person who is a teacher.
    iphone_user = nondet(solution.houses)
    assume(iphone_user.person.phone_model == "iphone 13")
    assert iphone_user.person.occupation == "teacher"
""",
            """struct Person {
    const char * name;
    const char * occupation;
    const char * book_genre;
    const char * phone_model;
    int age;
};

static const char * Person_name[] = {"Alice", "Eric", "Arnold", "Peter"};
static bool Person_name_used[4];
static const char * Person_occupation[] = {"artist", "engineer", "teacher", "doctor"};
static bool Person_occupation_used[4];
static const char * Person_book_genre[] = {"fantasy", "science fiction", "mystery", "romance"};
static bool Person_book_genre_used[4];
static const char * Person_phone_model[] = {"google pixel 6", "iphone 13", "oneplus 9", "samsung galaxy s21"};
static bool Person_phone_model_used[4];

static void init_Person(struct Person * instance) {
    __CPROVER_unique_domain(instance->name, Person_name);
    __CPROVER_unique_domain(instance->occupation, Person_occupation);
    __CPROVER_unique_domain(instance->book_genre, Person_book_genre);
    __CPROVER_unique_domain(instance->phone_model, Person_phone_model);
    instance->age = 10;
}

struct House {
    int id;
    struct Person person;
};

static int House_id[] = {1, 2, 3, 4};
static bool House_id_used[4];

static void init_House(struct House * instance) {
    __CPROVER_unique_domain(instance->id, House_id);
    init_Person(&instance->person);
}

struct Solution {
    struct House houses[4];
};

static void init_Solution(struct Solution * instance) {
    for (size_t i = 0; i < sizeof(instance->houses) / sizeof(instance->houses[0]); ++i) {
        init_House(&instance->houses[i]);
    }
}

""",
        )

    def __test_harness(self, code: str, expected_harness: str) -> None:
        source_tree: Module = parse_module(code)
        visitor = LogicPyCDataStructureGenerator(source_tree)
        source_tree.visit(visitor)
        print(visitor.c_harness)
        #self.assertEqual(expected_harness, visitor.c_harness)
