from ast import parse
from unittest import TestCase, main as unittest_main

from meta.asttools import cmp_ast, print_ast

from doctrans.info import parse_docstring
from doctrans.tests.mocks import cls, ast_def, docstring0


class TestParseDocstring(TestCase):
    maxDiff = 1651
    docstring0 = "\nLoad the data for your ML pipeline. Will be fed into `train`.\n\n" \
                 ":param dataset_name: name of dataset\n" \
                 ":type dataset_name: ```str```\n\n" \
                 ":param data_loader: function that returns the expected data type." \
                 "\n Defaults to TensorFlow Datasets and ml_prepare combined one.\n" \
                 ":type data_loader: ```Optional[(*args, **kwargs) -> Union[tf.data.Datasets, Any]]```\n\n" \
                 ":param data_loader_kwargs: pass this as arguments to data_loader function\n" \
                 ":type data_loader_kwargs: ```**data_loader_kwargs```\n\n" \
                 ":param data_type: incoming data type, defaults to 'infer'\n" \
                 ":type data_type: ```str```\n\n" \
                 ":param output_type: outgoing data_type, defaults to no conversion\n" \
                 ":type output_type: ```str```\n\n" \
                 ":param K: backend engine, e.g., `np` or `tf`\n" \
                 ":type K: ```Optional[Literal[np, tf]]```\n\n" \
                 ":return: Dataset splits (by default, your train and test)\n" \
                 ":rtype: ```Tuple[np.ndarray, np.ndarray]```\n"

    docstring1 = docstring0.replace(':type K', ':type notOK')

    def test_correctly_formatted(self) -> None:
        self.assertDictEqual(parse_docstring(self.docstring0),
                             {'long_description': '',
                              'params': [{'doc': 'name of dataset',
                                          'name': 'dataset_name',
                                          'typ': 'str'},
                                         {'doc': 'function that returns the expected data type.\n'
                                                 ' Defaults to TensorFlow Datasets and ml_prepare '
                                                 'combined one.',
                                          'name': 'data_loader',
                                          'typ': 'Optional[(*args, **kwargs) -> Union[tf.data.Datasets, Any]]'},
                                         {'doc': 'pass this as arguments to data_loader function',
                                          'name': 'data_loader_kwargs',
                                          'typ': '**data_loader_kwargs'},
                                         {'doc': "incoming data type, defaults to 'infer'",
                                          'name': 'data_type',
                                          'typ': 'str'},
                                         {'doc': 'outgoing data_type, defaults to no conversion',
                                          'name': 'output_type',
                                          'typ': 'str'},
                                         {'doc': 'backend engine, e.g., `np` or `tf`',
                                          'name': 'K',
                                          'typ': 'Optional[Literal[np, tf]]'}],
                              'returns': {'name': 'Dataset splits (by default, your train and test)',
                                          'typ': 'Tuple[np.ndarray, np.ndarray]'},
                              'short_description': 'Load the data for your ML pipeline. Will be fed into '
                                                   '`train`.'}
                             )

    def test_equality(self) -> None:
        # print_ast(parse(cls, mode='exec'))
        self.assertTrue(cmp_ast(parse(cls, mode='exec'), ast_def),
                        'class parsed as AST doesn\'t match constructed AST')
        self.assertDictEqual(parse_docstring(docstring0),
                             {'long_description': '',
                              'params': [{'doc': 'name of dataset',
                                          'name': 'dataset_name',
                                          'typ': 'str'},
                                         {'doc': 'directory to look for models in. Default is '
                                                 '~/tensorflow_datasets.',
                                          'name': 'tfds_dir',
                                          'typ': 'Optional[str]'},
                                         {'doc': 'backend engine, e.g., `np` or `tf`',
                                          'name': 'K',
                                          'typ': 'Optional[Literal[np, tf]]'},
                                         {'doc': 'Convert to numpy ndarrays',
                                          'name': 'as_numpy',
                                          'typ': 'bool'},
                                         {'doc': 'pass this as arguments to data_loader function',
                                          'name': 'data_loader_kwargs',
                                          'typ': '**data_loader_kwargs'}],
                              'returns': {'name': 'Train and tests dataset splits',
                                          'typ': 'Union[Tuple[tf.data.Dataset, tf.data.Dataset], '
                                                 'Tuple[np.ndarray, np.ndarray]]'},
                              'short_description': 'Acquire from the official tensorflow_datasets model '
                                                   'zoo, or the ophthalmology focussed ml-prepare '
                                                   'library'}
                             )

    def test_badly_formatted(self) -> None:
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(self.docstring1)
        self.assertEqual('\'K\' != \'notOK\'', cte.exception.__str__())


if __name__ == '__main__':
    unittest_main()
