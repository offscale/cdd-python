from unittest import TestCase, main as unittest_main

from docstring2class.info import parse_docstring


class TestParseDocstring(TestCase):
    docstring0 = "\nLoad the data for your ML pipeline. Will be fed into `train`." \
                 "\n\n:param dataset_name: name of dataset\n:type dataset_name: ```str```" \
                 "\n\n:param data_loader: function that returns the expected data type." \
                 "\n Defaults to TensorFlow Datasets and ml_prepare combined one." \
                 "\n:type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```" \
                 "\n\n:param data_loader_kwargs: pass this as arguments to data_loader function" \
                 "\n:type data_loader_kwargs: ```**data_loader_kwargs```" \
                 "\n\n:param data_type: incoming data type, defaults to 'infer'" \
                 "\n:type data_type: ```str```" \
                 "\n\n:param output_type: outgoing data_type, defaults to no conversion" \
                 "\n:type output_type: ```None or 'numpy'```" \
                 "\n\n:param K: backend engine, e.g., `np` or `tf`" \
                 "\n:type K: ```None or np or tf or Any```" \
                 "\n\n:return: Dataset splits (by default, your train and test)" \
                 "\n:rtype: ```Tuple[np.ndarray, np.ndarray]```" \
                 "\n"
    docstring1 = "\nLoad the data for your ML pipeline. Will be fed into `train`." \
                 "\n\n:param dataset_name: name of dataset\n:type dataset_name: ```str```" \
                 "\n\n:param data_loader: function that returns the expected data type." \
                 "\n Defaults to TensorFlow Datasets and ml_prepare combined one." \
                 "\n:type data_loader_wrong: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```" \
                 "\n\n:param data_loader_kwargs: pass this as arguments to data_loader function" \
                 "\n:type data_loader_kwargs: ```**data_loader_kwargs```" \
                 "\n\n:param data_type: incoming data type, defaults to 'infer'" \
                 "\n:type data_type: ```str```" \
                 "\n\n:param output_type: outgoing data_type, defaults to no conversion" \
                 "\n:type output_type: ```None or 'numpy'```" \
                 "\n\n:param K: backend engine, e.g., `np` or `tf`" \
                 "\n:type K: ```None or np or tf or Any```" \
                 "\n\n:return: Dataset splits (by default, your train and test)" \
                 "\n:rtype: ```Tuple[np.ndarray, np.ndarray]```" \
                 "\n"

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
                                          'typ': 'None or (*args, **kwargs) -> tf.data.Datasets or '
                                                 'Any'},
                                         {'doc': 'pass this as arguments to data_loader function',
                                          'name': 'data_loader_kwargs',
                                          'typ': '**data_loader_kwargs'},
                                         {'doc': "incoming data type, defaults to 'infer'",
                                          'name': 'data_type',
                                          'typ': 'str'},
                                         {'doc': 'outgoing data_type, defaults to no conversion',
                                          'name': 'output_type',
                                          'typ': "None or 'numpy'"},
                                         {'doc': 'backend engine, e.g., `np` or `tf`',
                                          'name': 'K',
                                          'typ': 'None or np or tf or Any'}],
                              'returns': '',
                              'short_description': 'Load the data for your ML pipeline. Will be fed into '
                                                   '`train`.'}
                             )

    def test_badly_formatted(self) -> None:
        with self.assertRaises(AssertionError) as cte:
            parse_docstring(self.docstring1)
        self.assertEqual('\'data_loader\' != \'data_loader_wrong\'', cte.exception.__str__())


if __name__ == '__main__':
    unittest_main()
