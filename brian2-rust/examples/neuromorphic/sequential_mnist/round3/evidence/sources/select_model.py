"""Select a frozen candidate using validation reports, without opening MNIST."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();candidates=[]
    for directory in a.runs:
        report=json.loads((directory/'training.json').read_text())
        assert report['test_arrays_read'] is False
        assert report['training_samples']==54000 and report['validation_samples']==6000
        assert report['updates_completed']==report['requested_updates'], 'wait for the candidate run to finish'
        assert sha(directory/'frozen.json')==report['frozen_file_sha256']
        assert sha(directory/'checkpoint.json')==report['checkpoint_sha256']
        candidates.append(dict(run=directory.name,validation_accuracy=report['selected']['validation_accuracy'],
            selected_step=report['selected']['step'],updates=report['updates_completed'],sizes=report['sizes'],
            encoding=report['encoding'],deskew=report.get('deskew',False),repeats=report['repeats'],
            recurrent_groups=report.get('recurrent_groups',1),active_weight_count=report.get('active_weight_count'),runner_sha256=report['runner_sha256'],
            query_ticks=report.get('query_ticks',8),steps=report['steps'],
            augmentation_shift=report.get('augmentation_shift',0),
            augmentation_affine=report.get('augmentation_affine',0.),
            augmentation_elastic=report.get('augmentation_elastic',0.),
            readout_start=report['readout_start'],history=report['history'],
            report_sha256=sha(directory/'training.json'),frozen_file_sha256=sha(directory/'frozen.json')))
    winner=max(range(len(candidates)),key=lambda i:candidates[i]['validation_accuracy'])
    source=a.runs[winner];a.output.mkdir(parents=True,exist_ok=False)
    for name in ('checkpoint.json','frozen.json','training.json','training-script.py'):
        if (source/name).exists():shutil.copy2(source/name,a.output/name)
    selected=json.loads((a.output/'training.json').read_text())
    assert sha(a.output/'frozen.json')==selected['frozen_file_sha256']
    assert sha(a.output/'checkpoint.json')==selected['checkpoint_sha256']
    result=dict(schema='atlas.sequential-mnist.selection.v1',selection='maximum validation accuracy; first candidate wins ties',
        test_data_read=False,selected=candidates[winner],candidates=candidates)
    (a.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
