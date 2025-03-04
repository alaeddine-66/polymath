# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from json import loads
from tempfile import NamedTemporaryFile
from unittest import IsolatedAsyncioTestCase, TestCase

import aiofiles

from agent.logic.zebra_benchmark import ZebraBenchmark

from dotenv import load_dotenv


load_dotenv()


class TestZebraBenchmark(IsolatedAsyncioTestCase):

    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.maxDiff = None

    def test_get_format(self):
        self.assertEqual(
            """{
    "solution": {
        "House 1": {
            "Name": "___",
            "Nationality": "___",
            "BookGenre": "___",
            "Food": "___",
            "Color": "___",
            "Animal": "___"
        },
        "House 2": {
            "Name": "___",
            "Nationality": "___",
            "BookGenre": "___",
            "Food": "___",
            "Color": "___",
            "Animal": "___"
        },
        "House 3": {
            "Name": "___",
            "Nationality": "___",
            "BookGenre": "___",
            "Food": "___",
            "Color": "___",
            "Animal": "___"
        },
        "House 4": {
            "Name": "___",
            "Nationality": "___",
            "BookGenre": "___",
            "Food": "___",
            "Color": "___",
            "Animal": "___"
        },
        "House 5": {
            "Name": "___",
            "Nationality": "___",
            "BookGenre": "___",
            "Food": "___",
            "Color": "___",
            "Animal": "___"
        }
    }
}""",
            ZebraBenchmark.get_format(
                loads(
                    '{ "header": [ "House", "Name", "Nationality", "BookGenre", "Food", "Color", "Animal" ], "rows": [ [ "XXX", "XXX", "XXX", "XXX", "XXX", "XXX", "XXX" ], [ "XXX", "XXX", "XXX", "XXX", "XXX", "XXX", "XXX" ], [ "XXX", "XXX", "XXX", "XXX", "XXX", "XXX", "XXX" ], [ "XXX", "XXX", "XXX", "XXX", "XXX", "XXX", "XXX" ], [ "XXX", "XXX", "XXX", "XXX", "XXX", "XXX", "XXX" ] ] }'
                )
            ),
        )

    async def test_end_to_end(self):
        with NamedTemporaryFile() as eval_json_file:
            eval_json_file.close()
            eval_json_file_name: str = eval_json_file.name
            zebraBenchmark = ZebraBenchmark(
                eval_json_file_name,
                "meta-llama/Meta-Llama-3.1-70B-Instruct@reasoning",
                "llama3-70b-instruct",
                False,
                lambda task: task["id"] == "lgp-test-5x6-16",
            )
            await zebraBenchmark.run()

            eval_json: str
            async with aiofiles.open(eval_json_file_name, "r") as file:
                eval_json = await file.read()
            self.assertTrue(
                r""""output": [
            "{\n    \"solution\": {\n        \"House 1\": {\n"""
                in eval_json
            )
