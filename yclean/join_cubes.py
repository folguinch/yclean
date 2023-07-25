"""Join cubes produced by yclean."""
from typing import Sequence, Optional, Callable
from pathlib import Path
import argparse
import sys
import os

from casatasks import exportfits
from casatools import image
from goco_helpers import argparse_actions as actions
from spectral_cube import SpectralCube
import matplotlib.pyplot as plt
import numpy as np

def crop_spectral_axis(img: image,
                       chans: str,
                       outfile: Path):
    """Crop image along the spectral axis.

    Args:
      img: CASA image object.
      chans: channel range.
      outfile: output image file.
    """
    # Identify spectral axis
    summ = img.summary()
    ind = np.where(summ['axisnames'] == 'Frequency')[0][0]

    # Crop image
    aux = img.crop(outfile=str(outfile), axes=ind, chans=chans)
    aux.close()

def join_cubes(inputs: Sequence[Path],
               output: Path,
               channels: Optional[Sequence[str]],
               export_fits: bool = False,
               resume: bool = False,
               log: Callable = print) -> Path:
    """Join cubes at specific channels.

    Args:
      inputs: input cubes to join.
      output: file name.
      channels: channel ranges for spectral cropping.
      resume: optional; resume calculations?
      export_fits: optional; export final image to fits (delete original)?
      log: optional; logging function.

    Returns:
      The filename of the final cube.
    """
    # Check
    if len(channels) != len(inputs):
        raise ValueError('Different length of input and channels')

    # Concatenated image
    imagename = output.expanduser()

    # Join
    has_temps = False
    if resume and imagename.exists():
        log(f'Skipping concatenated image: {imagename}')
    else:
        if imagename.exists():
            os.system(f'rm -rf {imagename}')
        # Crop images
        filelist = []
        for i, (chans, inp) in enumerate(zip(channels, inputs)):
            if chans is None:
                img_name = inp.expanduser()
            else:
                img = image()
                img = img.open(str(inp.expanduser()))
                img_name = Path(f'temp{i}.image')
                if img_name.is_dir():
                    os.system('rm -rf temp*.image')
                crop_spectral_axis(img, chans, img_name)
                img.close()
                has_temps = True

            # Store filenames
            filelist.append(str(img_name))
        filelist = ' '.join(filelist)

        # Concatenate
        img.imageconcat(outfile=str(imagename), infiles=filelist)
        img.close()

    # Export fits
    if export_fits:
        imagefits = imagename.with_suffix('.image.fits')
        if resume and imagefits.exists():
            log('Skipping FITS export')
        else:
            exportfits(imagename=str(imagename), fitsimage=str(imagefits),
                       overwrite=True)
            os.system(f'rm -rf {imagename}')
        imagename = imagefits

    # Clean up
    if has_temps:
        log('Cleaning up')
        os.system('rm -rf temp*.image')

    return imagename

def _join_cubes(args: NameSpace) -> None:
    """Join the cubes."""
    # Check if step is needed
    if args.chanranges is None:
        args.log.info('Cubes will not be cropped')
        args.chanranges = [None for v in range(len(args.cubes))]
    elif len(args.chanranges) != len(args.cubes):
        # Check lengths
        raise ValueError('Number of input cubes and chanranges do not match')

    # Join the cubes
    args.finalcube = join_cubes(args.cubes, args.outputcube,
                                channels=args.chanranges, export_fits=True,
                                log=args.log.info)

def _plot_spec(args: NameSpace) -> None:
    if args.spec_at is None:
        return
    spec_at = args.spec_at
    
    # Read cube
    cube = SpectralCube.read(args.finalcube)
    cube_spec = cube.unmasked_data[:, spec_at[1], spec_at[0]]

    # Some important values
    minval = np.nanmin(cube_spec[10:-10])
    minval = minval - (minval * 0.1)
    maxval = np.nanmax(cube_spec[10:-10])
    maxval = maxval + (maxval * 0.1)

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.set_xlim(0, cube_spec.size-1)
    ax.set_ylim(minval, maxval)
    ax.plot(cube_spec, 'k-', ds='steps-mid', zorder=1)
    fig.savefig(args.finalcube.with_suffix('.png'), bbox_inches='tight')

def _postproc(args: NameSpace) -> None:
    # Should the final cube be postprocessed?
    if args.common_beam or args.minimal:
        cube = SpectralCube.read(args.finalcube)

    # Minimal subcube
    if args.minimal:
        cube = cube.minimal_subcube()

    # Convolve with common beam
    if args.common_beam:
        common_beam = cube.beams.common_beam()
        cube = cube.convolve_to(common_beam)

    # Save the final cube
    cube.write(args.finalcube)

def join_cubes_cmd(args: Optional[Sequence] = None) -> None:
    """Join cubes from command line inputs.

    Args:
      args: command line args.
    """
    # Pipe
    pipe=[_join_cubes, _plot_spec, _postproc]

    # Command line options
    logfile = datetime.now().isoformat(timespec='milliseconds')
    logfile = f'debug_joincubes_{logfile}.log'
    args_parents = [parents.logger(logfile)]
    parser = argparse.ArgumentParser(
        description='Join cubes from command line inputs',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=args_parents,
        conflict_handler='resolve',
    )
    parser.add_argument('--common_beam', action='store_true',
                        help='Compute the common beam of the final cube')
    parser.add_argument('--minimal', action='store_true',
                        help='Compute the minimal subcube')
    parser.add_argument('--spec_at', metavar=('XPOS', 'YPOS'), type=int,
                        nargs=2, default=None,
                        help='Plot steps at this pixel position')
    parser.add_argument('--chanranges', args='*',
                        help='Channel ranges to crop the cubes')
    parser.add_argument('outputcube', nargs='1', type=str,
                        action=actions.NormalizePath,
                        help='Output cube path')
    parser.add_argument('cubes', nargs='*', type=str,
                        action=actions.NormalizePath,
                        help='Cubes to join')
    parser.set_defaults(finalcube=None)
    # Check args
    if args is None:
        args = sys.argv[1:]
    args = parser.parse_args(args)

    # Run steps
    for step in pipe:
        step(args)

if __name__ == '__main__':
    join_cubes_main(sys.argv[1:])
