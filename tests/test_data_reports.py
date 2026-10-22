import csv
import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image
from image_classifier.data import CLASSES, PREPROCESS, load_rgb, verify_dataset
from image_classifier.reports import evaluation_report
from image_classifier.metrics import classification_metrics
from image_classifier.checkpoint import atomic_save, load_model


def fixture_data(root):
    (root/'splits').mkdir()
    (root/'class_to_idx.json').write_text(json.dumps(CLASSES))
    for split in ('train', 'test'):
        rows=[]
        for i,label in enumerate(CLASSES):
            p=root/split/label/'sample.png'
            p.parent.mkdir(parents=True)
            Image.new('RGB',(4,4),(i*60, 10 if split=='train' else 30, 1)).save(p)
            original=root/'images'/label/f'{split}.png'
            original.parent.mkdir(parents=True,exist_ok=True)
            original.write_bytes(p.read_bytes())
            rows.append({'image_id':split+label,'label':label,'path':str(original.relative_to(root)),
                         'split':split,'split_group':split+label,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
            # Manifest basename must equal the physical split basename.
            original.rename(original.with_name('sample_'+split+'.png'))
            p.rename(p.with_name('sample_'+split+'.png'))
            rows[-1]['path']=str(original.with_name('sample_'+split+'.png').relative_to(root))
        with (root/'splits'/f'{split}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0])
            writer.writeheader();writer.writerows(rows)


def test_verifier_rejects_missing_and_altered_images(tmp_path):
    fixture_data(tmp_path)
    rows=verify_dataset(tmp_path)
    p=tmp_path/rows['train'][0]['file']
    data=p.read_bytes()
    p.unlink()
    with pytest.raises(ValueError,match='Missing/unreadable image'):
        verify_dataset(tmp_path)
    p.write_bytes(data+b'changed')
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_dataset(tmp_path)


def test_alpha_white(tmp_path):
    p=tmp_path/'alpha.png'
    Image.new('RGBA',(2,2),(255,0,0,0)).save(p)
    assert load_rgb(p).getpixel((0,0))==(255,255,255)


def test_report_counts_and_gallery(tmp_path):
    fixture_data(tmp_path)
    rows=verify_dataset(tmp_path)['test']
    for row in rows:
        row.update(source='fixture',subtype='style',source_work='work')
    targets,predicted=[0,1,2],[0,0,2]
    scores=[[.8,.1,.1],[.6,.3,.1],[.1,.1,.8]]
    output=tmp_path/'report'
    evaluation_report(tmp_path,rows,classification_metrics(targets,predicted),
                      {'targets':targets,'predictions':predicted,'scores':scores},output,'fixture.pt')
    report=json.loads((output/'metrics.json').read_text())
    assert report['by_source']['fixture']['count']==3
    assert len(list((output/'mistakes').glob('*.jpg')))==1
    assert len(list(csv.DictReader((output/'predictions.csv').open())))==3


def test_reject_incompatible_checkpoint(tmp_path):
    path=tmp_path/'bad.pt'
    atomic_save(path,{'architecture':'convnext_tiny','class_to_idx':{'live':0},'preprocessing':PREPROCESS})
    with pytest.raises(ValueError,match='incompatible'):
        load_model(path,'cpu')
