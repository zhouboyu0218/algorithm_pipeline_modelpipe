# Copyright (c) OpenMMLab. All rights reserved.
import ast
import os.path as osp
import sys
import warnings
import importlib
from collections import defaultdict
from importlib.util import find_spec
from collections import abc
from typing import List, Any, Optional, Type, Union
from importlib import import_module
from inspect import ismodule

PYTHON_ROOT_DIR = osp.dirname(osp.dirname(sys.executable))


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def package2module(package: str):
    """Infer module name from package.

    Args:
        package (str): Package to infer module name.
    """
    from pkg_resources import get_distribution
    pkg = get_distribution(package)
    if pkg.has_metadata('top_level.txt'):
        module_name = pkg.get_metadata('top_level.txt').split('\n')[0]
        return module_name
    else:
        raise ValueError(f'can not infer the module name of {package}')


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Args:
        package (str): Name of package.

    Example:
        >>> get_installed_path('mmcls')
        >>> '.../lib/python3.7/site-packages/mmcls'
    """
    from pkg_resources import get_distribution

    # if the package name is not the same as module name, module name should be
    # inferred. For example, mmcv-full is the package name, but mmcv is module
    # name. If we want to get the installed path of mmcv-full, we should concat
    # the pkg.location and module name
    pkg = get_distribution(package)
    possible_path = osp.join(pkg.location, package)
    if osp.exists(possible_path):
        return possible_path
    else:
        return osp.join(pkg.location, package2module(package))


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Args:
        package (str): Name of package to be checked.
    """
    # When executing `import mmengine.runner`,
    # pkg_resources will be imported and it takes too much time.
    # Therefore, import it in function scope to save time.
    import pkg_resources
    from pkg_resources import get_distribution

    # refresh the pkg_resources
    # more datails at https://github.com/pypa/setuptools/issues/373
    importlib.reload(pkg_resources)
    try:
        get_distribution(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False


MODULE2PACKAGE = {
    'mmcls': 'mmcls',
    'mmdet': 'mmdet',
    'mmdet3d': 'mmdet3d',
    'mmseg': 'mmsegmentation',
    'mmaction': 'mmaction2',
    'mmtrack': 'mmtrack',
    'mmpose': 'mmpose',
    'mmedit': 'mmedit',
    'mmocr': 'mmocr',
    'mmgen': 'mmgen',
    'mmfewshot': 'mmfewshot',
    'mmrazor': 'mmrazor',
    'mmflow': 'mmflow',
    'mmhuman3d': 'mmhuman3d',
    'mmrotate': 'mmrotate',
    'mmselfsup': 'mmselfsup',
    'mmyolo': 'mmyolo',
    'mmpretrain': 'mmpretrain',
    'mmagic': 'mmagic',
}

# PKG2PROJECT is not a proper name to represent the mapping between module name
# (module import from) and package name (used by pip install). Therefore,
# PKG2PROJECT will be deprecated and this alias will only be kept until
# MMEngine v1.0.0
PKG2PROJECT = MODULE2PACKAGE


class ConfigParsingError(RuntimeError):
    """Raise error when failed to parse pure Python style config files."""


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node


def _is_builtin_module(module_name: str) -> bool:
    """Check if a module is a built-in module.

    Arg:
        module_name: name of module.
    """
    if module_name.startswith('.'):
        return False
    if module_name.startswith('mmengine.config'):
        return True
    spec = find_spec(module_name.split('.')[0])
    # Module not found
    if spec is None:
        return False
    origin_path = getattr(spec, 'origin', None)
    if origin_path is None:
        return False
    origin_path = osp.abspath(origin_path)
    if ('site-package' in origin_path
            or not origin_path.startswith(PYTHON_ROOT_DIR)):
        return False
    else:
        return True


class ImportTransformer(ast.NodeTransformer):
    """Convert the import syntax to the assignment of
    :class:`mmengine.config.LazyObject` and preload the base variable before
    parsing the configuration file.

    Since you are already looking at this part of the code, I believe you must
    be interested in the mechanism of the ``lazy_import`` feature of
    :class:`Config`. In this docstring, we will dive deeper into its
    principles.

    Most of OpenMMLab users maybe bothered with that:

        * In most of popular IDEs, they cannot navigate to the source code in
          configuration file
        * In most of popular IDEs, they cannot jump to the base file in current
          configuration file, which is much painful when the inheritance
          relationship is complex.

    In order to solve this problem, we introduce the ``lazy_import`` mode.

    A very intuitive idea for solving this problem is to import the module
    corresponding to the "type" field using the ``import`` syntax. Similarly,
    we can also ``import`` base file.

    However, this approach has a significant drawback. It requires triggering
    the import logic to parse the configuration file, which can be
    time-consuming. Additionally, it implies downloading numerous dependencies
    solely for the purpose of parsing the configuration file.
    However, it's possible that only a portion of the config will actually be
    used. For instance, the package used in the ``train_pipeline`` may not
    be necessary for an evaluation task. Forcing users to download these
    unused packages is not a desirable solution.

    To avoid this problem, we introduce :class:`mmengine.config.LazyObject` and
    :class:`mmengine.config.LazyAttr`. Before we proceed with further
    explanations, you may refer to the documentation of these two modules to
    gain an understanding of their functionalities.

    Actually, one of the functions of ``ImportTransformer`` is to hack the
    ``import`` syntax. It will replace the import syntax
    (exclude import the base files) with the assignment of ``LazyObject``.

    As for the import syntax of the base file, we cannot lazy import it since
    we're eager to merge the fields of current file and base files. Therefore,
    another function of the ``ImportTransformer`` is to collaborate with
    ``Config._parse_lazy_import`` to parse the base files.

    Args:
        global_dict (dict): The global dict of the current configuration file.
            If we divide ordinary Python syntax into two parts, namely the
            import section and the non-import section (assuming a simple case
            with imports at the beginning and the rest of the code following),
            the variables generated by the import statements are stored in
            global variables for subsequent code use. In this context,
            the ``global_dict`` represents the global variables required when
            executing the non-import code. ``global_dict`` will be filled
            during visiting the parsed code.
        base_dict (dict): All variables defined in base files.

            Examples:
                >>> from mmengine.config import read_base
                >>>
                >>>
                >>> with read_base():
                >>>     from .._base_.default_runtime import *
                >>>     from .._base_.datasets.coco_detection import dataset

            In this case, the base_dict will be:

            Examples:
                >>> base_dict = {
                >>>     '.._base_.default_runtime': ...
                >>>     '.._base_.datasets.coco_detection': dataset}

            and `global_dict` will be updated like this:

            Examples:
                >>> global_dict.update(base_dict['.._base_.default_runtime'])  # `import *` means update all data
                >>> global_dict.update(dataset=base_dict['.._base_.datasets.coco_detection']['dataset'])  # only update `dataset`
    """  # noqa: E501

    def __init__(self,
                 global_dict: dict,
                 base_dict: Optional[dict] = None,
                 filename: Optional[str] = None):
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        # In Windows, the filename could be like this:
        # "C:\\Users\\runneradmin\\AppData\\Local\\"
        # Although it has been an raw string, ast.parse will firstly escape
        # it as the executed code:
        # "C:\Users\runneradmin\AppData\Local\\\"
        # As you see, the `\U` will be treated as a part of
        # the escape sequence during code parsing, leading to an
        # parsing error
        # Here we use `encode('unicode_escape').decode()` for double escaping
        if isinstance(filename, str):
            filename = filename.encode('unicode_escape').decode()
        self.filename = filename
        self.imported_obj: set = set()
        super().__init__()

    def visit_ImportFrom(
        self, node: ast.ImportFrom
    ) -> Optional[Union[List[ast.Assign], ast.ImportFrom]]:
        """Hack the ``from ... import ...`` syntax and update the global_dict.

        Examples:
            >>> from mmdet.models import RetinaNet

        Will be parsed as:

        Examples:
            >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')

        ``global_dict`` will also be updated by ``base_dict`` as the
        class docstring says.

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            Optional[List[ast.Assign]]: There three cases:

                * If the node is a statement of importing base files.
                  None will be returned.
                * If the node is a statement of importing a builtin module,
                  node will be directly returned
                * Otherwise, it will return the assignment statements of
                  ``LazyObject``.
        """
        # Built-in modules will not be parsed as LazyObject
        module = f'{node.level*"."}{node.module}'
        if _is_builtin_module(module):
            return node

        if module in self.base_dict:
            for alias_node in node.names:
                if alias_node.name == '*':
                    self.global_dict.update(self.base_dict[module])
                    return None
                if alias_node.asname is not None:
                    base_key = alias_node.asname
                else:
                    base_key = alias_node.name
                self.global_dict[base_key] = self.base_dict[module][
                    alias_node.name]
            return None

        nodes: List[ast.Assign] = []
        for alias_node in node.names:
            # `ast.alias` has lineno attr after Python 3.10,
            if hasattr(alias_node, 'lineno'):
                lineno = alias_node.lineno
            else:
                lineno = node.lineno
            if alias_node.name == '*':
                # TODO: If users import * from a non-config module, it should
                # fallback to import the real module and raise a warning to
                # remind users the real module will be imported which will slow
                # down the parsing speed.
                raise ConfigParsingError(
                    'Illegal syntax in config! `from xxx import *` is not '
                    'allowed to appear outside the `if base:` statement')
            elif alias_node.asname is not None:
                # case1:
                # from mmengine.dataset import BaseDataset as Dataset ->
                # Dataset = LazyObject('mmengine.dataset', 'BaseDataset')
                code = f'{alias_node.asname} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.asname)
            else:
                # case2:
                # from mmengine.model import BaseModel
                # BaseModel = LazyObject('mmengine.model', 'BaseModel')
                code = f'{alias_node.name} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.name)
            try:
                nodes.append(ast.parse(code).body[0])  # type: ignore
            except Exception as e:
                raise ConfigParsingError(
                    f'Cannot import {alias_node} from {module}'
                    '1. Cannot import * from 3rd party lib in the config '
                    'file\n'
                    '2. Please check if the module is a base config which '
                    'should be added to `_base_`\n') from e
        return nodes

    def visit_Import(self, node) -> Union[ast.Assign, ast.Import]:
        """Work with ``_gather_abs_import_lazyobj`` to hack the ``import ...``
        syntax.

        Examples:
            >>> import mmcls.models
            >>> import mmcls.datasets
            >>> import mmcls

        Will be parsed as:

        Examples:
            >>> # import mmcls.models; import mmcls.datasets; import mmcls
            >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

        Args:
            node (ast.AST): The node of the current import statement.

        Returns:
            ast.Assign: If the import statement is ``import ... as ...``,
            ast.Assign will be returned, otherwise node will be directly
            returned.
        """
        # For absolute import like: `import mmdet.configs as configs`.
        # It will be parsed as:
        # configs = LazyObject('mmdet.configs')
        # For absolute import like:
        # `import mmdet.configs`
        # `import mmdet.configs.default_runtime`
        # This will be parsed as
        # mmdet = LazyObject(['mmdet.configs.default_runtime', 'mmdet.configs])
        # However, visit_Import cannot gather other import information, so
        # `_gather_abs_import_LazyObject` will gather all import information
        # from the same module and construct the LazyObject.
        alias_list = node.names
        assert len(alias_list) == 1, (
            'Illegal syntax in config! import multiple modules in one line is '
            'not supported')
        # TODO Support multiline import
        alias = alias_list[0]
        if alias.asname is not None:
            self.imported_obj.add(alias.asname)
            return ast.parse(  # type: ignore
                f'{alias.asname} = LazyObject('
                f'"{alias.name}",'
                f'location="{self.filename}, line {node.lineno}")').body[0]
        return node


def _gather_abs_import_lazyobj(tree: ast.Module,
                               filename: Optional[str] = None):
    """Experimental implementation of gathering absolute import information."""
    if isinstance(filename, str):
        filename = filename.encode('unicode_escape').decode()
    imported = defaultdict(list)
    abs_imported = set()
    new_body: List[ast.stmt] = []
    # module2node is used to get lineno when Python < 3.10
    module2node: dict = dict()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip converting built-in module to LazyObject
                if _is_builtin_module(alias.name):
                    new_body.append(node)
                    continue
                module = alias.name.split('.')[0]
                module2node.setdefault(module, node)
                imported[module].append(alias)
            continue
        new_body.append(node)

    for key, value in imported.items():
        names = [_value.name for _value in value]
        if hasattr(value[0], 'lineno'):
            lineno = value[0].lineno
        else:
            lineno = module2node[key].lineno
        lazy_module_assign = ast.parse(
            f'{key} = LazyObject({names}, location="{filename}, line {lineno}")'  # noqa: E501
        )  # noqa: E501
        abs_imported.add(key)
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree, abs_imported


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def get_object_from_string(obj_name: str):
    """Get object from name.

    Args:
        obj_name (str): The name of the object.

    Examples:
        >>> get_object_from_string('torch.optim.sgd.SGD')
        >>> torch.optim.sgd.SGD
    """
    parts = iter(obj_name.split('.'))
    module_name = next(parts)
    # import module
    while True:
        try:
            module = import_module(module_name)
            part = next(parts)
            # mmcv.ops has nms.py has nms function at the same time. So the
            # function will have a higher priority
            obj = getattr(module, part, None)
            if obj is not None and not ismodule(obj):
                break
            module_name = f'{module_name}.{part}'
        except StopIteration:
            # if obj is a module
            return module
        except ImportError:
            return None

    # get class or attribute from module
    while True:
        try:
            obj_cls = getattr(module, part)
            part = next(parts)
        except StopIteration:
            return obj_cls
        except AttributeError:
            return None
