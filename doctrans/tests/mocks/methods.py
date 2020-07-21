from doctrans.tests.mocks.docstrings import docstring_structure_no_default_doc

docstring_structure_no_default_doc
class F(object):
    # {docstring_structure_no_default_doc}

    def method(self, dataset_name, tfds_dir, K, as_numpy, data_loader_kwargs):
        return (np.empty(0), np.empty(0))
