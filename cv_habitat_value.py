"""CV hab value."""
import argparse
import logging
import os
import sys

from osgeo import gdal
from osgeo import osr
import pygeoprocessing
import numpy
import taskgraph


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.INFO)


def _mask_by_nodata_op(
        base_array, mask_array, mask_nodata_value, target_nodata_value):
    """Set base == target_nodata when mask == mask_nodata."""
    result = numpy.array(base_array)
    nodata_mask = numpy.isclose(mask_array, mask_nodata_value)
    result[nodata_mask] = target_nodata_value
    return result


def mask_by_nodata(
        base_raster_path, nodata_mask_raster_path, target_raster_path):
    """Set base to nodata wherever mask is nodata."""
    base_raster_info = pygeoprocessing.get_raster_info(base_raster_path)
    nodata_mask_raster_info = pygeoprocessing.get_raster_info(
        nodata_mask_raster_path)
    pygeoprocessing.raster_calculator(
        [(base_raster_path, 1),
         (nodata_mask_raster_path, 1),
         (nodata_mask_raster_info['nodata'][0], 'raw'),
         (base_raster_info['nodata'][0], 'raw')], _mask_by_nodata_op,
        target_raster_path, base_raster_info['datatype'],
        base_raster_info['nodata'][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CV habitat value utility')
    parser.add_argument(
        '--shoreline_point_vector_path', type=str, required=True,
        help='path to point vector representing shore points')
    parser.add_argument(
        '--shoreline_vector_fieldname', type=str, required=True,
        help='fieldname for the quantity to project to the habitat')
    parser.add_argument(
        '--habitat_vector_path', type=str, required=True,
        help='path to polygon vector showing habitat coverage')
    parser.add_argument(
        '--protective_distance', type=float, required=True,
        help=(
            'maximum protective distance of the habitat layer in the same '
            'units as `shoreline_vector_fieldname` or the projection given '
            'as `target_epsg_code`.'))
    parser.add_argument(
        '--aoi_mask_raster_path', type=str, required=True,
        help='any nodata are masked out of the result')
    parser.add_argument(
        '--target_epsg_code', type=str,
        help=(
            'if defined the spatial analysis is done in this coordinate '
            'reference system, if not the coordinate reference system is '
            'assumed to be that of `habitat_vector_path`'))
    parser.add_argument(
        '--target_pixel_length', type=float, help=(
            'the side length of a target pixel in `target_epsg_code` units, '
            'or if not defined, `aoi_mask_raster_path`.'))
    parser.add_argument(
        '--target_habitat_value_raster_filename', type=str, required=True,
        help=(
            'desired filename path for the habitat value raster in the '
            'workspace directory'))
    parser.add_argument(
        '--target_shoreline_raster_filename', type=str,
        default='shoreline_raster.tif', help=(
            'if present this the shoreline point vector is rasterized to '
            'this filename in the workspace directory.'))

    parser.add_argument(
        '--workspace_dir', type=str, default='workspace', help=(
            'set the workspace for intermediate and target files, defaults '
            'to "./workspace"'))

    args = parser.parse_args()

    churn_dir = os.path.join(args.workspace_dir, 'churn')
    try:
        os.makedirs(churn_dir)
    except OSError:
        pass

    # ensure AOI and CV points are in the same projection
    aoi_raster_info = pygeoprocessing.get_raster_info(
        args.aoi_mask_raster_path)

    aoi_srs = osr.SpatialReference()
    aoi_srs.ImportFromWkt(aoi_raster_info['projection_wkt'])
    aoi_epsg = aoi_srs.GetAttrValue("PROJCS|GEOGCS|AUTHORITY", 1)

    shoreline_point_info = pygeoprocessing.get_vector_info(
        args.shoreline_point_vector_path)
    shoreline_srs = osr.SpatialReference()
    shoreline_srs.ImportFromWkt(shoreline_point_info['projection_wkt'])
    shoreline_epsg = aoi_srs.GetAttrValue("PROJCS|GEOGCS|AUTHORITY", 1)

    habitat_vector_info = pygeoprocessing.get_vector_info(
        args.habitat_vector_path)
    habitat_vector_srs = osr.SpatialReference()
    habitat_vector_srs.ImportFromWkt(habitat_vector_info['projection_wkt'])
    habitat_vector_epsg = aoi_srs.GetAttrValue("PROJCS|GEOGCS|AUTHORITY", 1)

    if len(set([habitat_vector_epsg, shoreline_epsg, aoi_epsg])) > 1:
        raise ValueError(
            "AOI raster, shoreline point vector, and habitat vector do not "
            "all share the same  projection")

    # Rasterize CV points w/ value using target AOI as mask
    pre_mask_point_raster_path = os.path.join(
        args.workspace_dir, 'pre_mask_shore_points.tif')
    target_pixel_size = aoi_raster_info['pixel_size']
    pygeoprocessing.new_raster_from_base(
        args.aoi_mask_raster_path, pre_mask_point_raster_path,
        gdal.GDT_Float32, [numpy.finfo(numpy.float32).min])
    pygeoprocessing.rasterize(
        args.shoreline_point_vector_path, pre_mask_point_raster_path,
        option_list=[
            f'ATTRIBUTE={args.shoreline_vector_fieldname}',
            'ALL_TOUCHED=TRUE'])

    # TODO: mask out values that are not in a defined AOI.
    shore_point_raster_path = os.path.join(
        args.workspace_dir, args.target_shoreline_raster_filename)
    mask_by_nodata(
        pre_mask_point_raster_path, args.aoi_mask_raster_path,
        shore_point_raster_path)

    # Create habitat mask
    habitat_mask_raster_path = os.path.join(
        args.workspace_dir, 'habitat_mask.tif')
    pygeoprocessing.new_raster_from_base(
        args.aoi_mask_raster_path, habitat_mask_raster_path,
        gdal.GDT_Byte, [0])
    pygeoprocessing.rasterize(
        args.habitat_vector_path, habitat_mask_raster_path,
        burn_values=[1], option_list=['ALL_TOUCHED=TRUE'])

    # Make convolution kernel
    kernel_path = os.path.join(churn_dir, 'kernel.tif')
    # assume square pixels
    kernel_radius = int(args.protective_distance // target_pixel_size[0])
    LOGGER.info(f"kernel radius: {kernel_radius}")
    kernel_x, kernel_y = numpy.meshgrid(
        range((kernel_radius-1)*2+1),
        range((kernel_radius-1)*2+1))
    kernel_distance = numpy.sqrt(
        (kernel_x-(kernel_radius-1))**2 +
        (kernel_y-(kernel_radius-1))**2)
    kernel_array = (kernel_distance <= kernel_radius).astype(numpy.int8)

    pygeoprocessing.numpy_array_to_raster(
        kernel_array, 0, (1, -1), (0, 0), None, kernel_path)

    # Convolve CV points for coverage
    convolve_target_raster_path = os.path.join(churn_dir, 'convolve_2d.tif')
    pygeoprocessing.convolve_2d(
        (shore_point_raster_path, 1), (kernel_path, 1),
        convolve_target_raster_path, ignore_nodata_and_edges=False,
        mask_nodata=False, normalize_kernel=False,
        target_datatype=gdal.GDT_Float64)

    target_habitat_value_raster_path = os.path.join(
        args.workspace_dir, args.target_habitat_value_raster_filename)

    # TODO: mask result to habitat
    mask_by_nodata(
        convolve_target_raster_path, habitat_mask_raster_path,
        target_habitat_value_raster_path)
