"""CPU-only compact export outside the immutable159 numerical source package."""
from pathlib import Path
import copy
import json
import shutil
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont,ImageOps
from current.progress import track,stage
from refnet.protocol import read,write,sha,fp
from refnet_f1.protocol import inputs,verify_file,guard_space
from refnet_ablation.protocol import home,case_paths,baseline_location,summarize


def selection(names,scene,seed,policy):
    if len(set(names))!=len(names):raise ValueError('Duplicate official test image names')
    return sorted(names,key=lambda name:fp([policy['selection_salt'],scene,seed,name]))[:policy['pictures_per_case']]


def font(size):
    for name in ['DejaVuSans.ttf','arial.ttf']:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def targets(xml,classes):
    rows=[]
    for obj in ET.parse(xml).getroot().findall('object'):
        if int(obj.findtext('difficult','0')):continue
        name=obj.findtext('name')
        rows.append(dict(box=[float(obj.find('bndbox').findtext(k))-1 for k in ['xmin','ymin','xmax','ymax']],
            label=('K: ' if name in classes else 'U: ')+name,unknown=name not in classes))
    return rows


def draw_panel(image,rows,width,height,prediction=False):
    scale=min((width-20)/image.width,(height-20)/image.height)
    size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    panel=Image.new('RGB',(width,height),(235,237,240));x=(width-size[0])//2;y=(height-size[1])//2
    panel.paste(image.resize(size,Image.Resampling.LANCZOS),(x,y));draw=ImageDraw.Draw(panel);f=font(13)
    for row in rows:
        a,b,c,d=row['box'];a=max(x,min(x+size[0]-1,x+a*scale));c=max(a,min(x+size[0]-1,x+c*scale))
        b=max(y,min(y+size[1]-1,y+b*scale));d=max(b,min(y+size[1]-1,y+d*scale))
        color=(220,55,45) if row['unknown'] and prediction else (205,145,10) if row['unknown'] else (0,125,165) if prediction else (25,140,70)
        draw.rectangle((a,b,c,d),outline=color,width=2)
        text=row['label'];bounds=draw.textbbox((0,0),text,font=f);tw=min(width-4,bounds[2]-bounds[0]+6);th=bounds[3]-bounds[1]+6
        tx=max(1,min(width-tw-1,a));ty=max(1,min(height-th-1,b-th))
        draw.rectangle((tx,ty,tx+tw,ty+th),fill=color);draw.text((tx+3,ty+1),text,fill='white',font=f)
    return panel


def render(image_path,xml,prediction,classes,title,dest,policy):
    with Image.open(image_path) as raw:image=ImageOps.exif_transpose(raw).convert('RGB')
    gt=targets(xml,classes)
    if not len(prediction['boxes'])==len(prediction['scores'])==len(prediction['labels']):raise ValueError('Misaligned sampled prediction')
    order=sorted(range(len(prediction['scores'])),key=lambda i:(-prediction['scores'][i],i))[:policy['max_drawn_predictions']]
    rows=[]
    for i in order:
        label=int(prediction['labels'][i]);unknown=label in [-1,len(classes)]
        if not unknown and not 0<=label<len(classes):raise ValueError('Invalid sampled class index')
        name='unknown' if unknown else classes[label]
        rows.append(dict(box=prediction['boxes'][i],label=f'{name} {prediction["scores"][i]:.3f}',unknown=unknown))
    width,height=policy['panel_width'],policy['panel_height'];canvas=Image.new('RGB',(width*2,height+106),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),title,fill=(25,35,45),font=font(16))
    draw.text((12,35),'Ground truth: green known / amber unknown; difficult omitted',fill=(30,40,50),font=font(13))
    draw.text((width+12,35),f'F1 predictions: showing top {len(rows)} / {len(prediction["boxes"])}; cyan known / red unknown',fill=(30,40,50),font=font(13))
    canvas.paste(draw_panel(image,gt,width,height),(0,62));canvas.paste(draw_panel(image,rows,width,height,True),(width,62))
    draw.text((12,height+72),'Qualitative sample only. Reported AP uses every official test image and all exported detections.',fill=(70,75,80),font=font(13))
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.jpg.tmp')
    canvas.save(tmp,format='JPEG',quality=policy['jpeg_quality'],optimize=True);tmp.replace(dest)


def baseline_data_binding(root,cfg,scene,seed,dataset_sha):
    folder=baseline_location(root,cfg,scene,seed)
    path=folder/'LOCK.json'
    if scene=='Game':
        entry=folder/'CORRECTED_PARENT.json'
        if not entry.exists():raise RuntimeError('Corrected Game full-control metadata missing for compact export')
        parent=read(entry);path=Path(root)/parent['lock']
    binding=read(path)
    if binding['annotation']['dataset_manifest_sha256']!=dataset_sha:
        raise RuntimeError('Cannot draw current GT against metrics from another dataset revision')


def add_file(entries,path,arc=None):
    path=Path(path)
    if not path.is_file():return
    name=str(arc).replace('\\','/') if arc is not None else path.as_posix()
    if name in entries and entries[name]!=path:raise RuntimeError('Duplicate compact archive member: '+name)
    entries[name]=path


def export_results(root,cfg):
    root=Path(root);folder=home(root,cfg);policy=read(root/'configs/reference_ablation_export.json')
    if policy['pictures_per_case']!=10:raise ValueError('This compact export fixes ten images per evaluated scene/seed/configuration')
    from refnet_ablation.localization import summarize as summarize_localization
    core=summarize(root,cfg);diag=summarize_localization(root,cfg)
    staging=folder/'COMPACT_EXPORT';staging.mkdir(parents=True,exist_ok=True)
    previous=read(staging/'IMAGE_SELECTION.json') if (staging/'IMAGE_SELECTION.json').exists() else dict(images=[])
    previous={r['archive_path']:r for r in previous['images']};entries={};image_rows=[];cases=[];missing=[];verified={}
    dataset_path=root/'assets/rwd/DATASET_MANIFEST.json';dataset=read(dataset_path);dataset_sha=sha(dataset_path)
    exporter_source={name:sha(root/name) for name in ['run_reference_ablation_compact.py','src/reference_ablation_export.py','configs/reference_ablation_export.json']}
    for name in ['CORE_SUMMARY.json','CORE_SUMMARY.md','LOCALIZATION_SUMMARY.json','LOCALIZATION_SUMMARY.md']:
        add_file(entries,folder/name,(folder/name).relative_to(root).as_posix())
    for scene in cfg['scenes']:
        for seed in cfg['seeds']:
            spec,_,_=inputs(root,scene,'task1',seed)
            selected=selection(spec['test_images'],scene,seed,policy)
            kinds=(['full'] if policy['include_full_controls'] else [])+cfg['variants']
            for variant in track(kinds,f'159 compact {scene}/seed_{seed}',unit='configurations'):
                report=baseline_location(root,cfg,scene,seed) if variant=='full' else case_paths(root,cfg,variant,scene,seed)[1]
                result_path=report/'RESULT.json';metric_path=report/'F1/metrics.json';prediction_path=report/'F1/predictions.json'
                key=f'{variant}/{scene}/seed_{seed}'
                if not result_path.exists() or not metric_path.exists():missing.append(key);continue
                result=read(result_path)
                if not result.get('complete'):missing.append(key);continue
                if (result['scene'],result['task'],result['seed'])!=(scene,'task1',seed):raise RuntimeError('Compact case identity mismatch: '+key)
                if variant=='full' and scene=='Game' and not result.get('corrected_label_retraining'):raise RuntimeError('Never substitute historical Game control')
                baseline_data_binding(root,cfg,scene,seed,dataset_sha)
                expected=result['artifacts']['F1/metrics.json'];verify_file(metric_path,expected,verified)
                metrics=read(metric_path)
                expected_predictions=result['artifacts']['F1/predictions.json'];verify_file(prediction_path,expected_predictions,verified)
                if metrics['prediction_sha256']!=expected_predictions:raise RuntimeError('Metric/prediction binding differs: '+key)
                predictions=read(prediction_path)
                if len(predictions)!=len(spec['test_images']) or len({p['image'] for p in predictions})!=len(predictions) or {p['image'] for p in predictions}!=set(spec['test_images']):
                    raise RuntimeError('Compact export requires sealed complete test predictions')
                predictions={p['image']:p for p in predictions}
                add_file(entries,metric_path,metric_path.relative_to(root).as_posix())
                case_images=[]
                for index,name in enumerate(track(selected,f'{key} render fixed examples',unit='images'),1):
                    image=root/'assets/rwd/JPEGImages'/scene/name;xml=root/'assets/rwd/Annotations'/scene/(Path(name).stem+'.xml')
                    image_sha=verify_file(image,dataset['files'][f'JPEGImages/{scene}/{name}']['sha256'],verified)
                    xml_sha=verify_file(xml,dataset['files'][f'Annotations/{scene}/{Path(name).stem}.xml']['sha256'],verified)
                    arc=(folder/'examples'/scene/f'seed_{seed}'/variant/f'{index:02d}_{fp(name)[:12]}.jpg').relative_to(root).as_posix()
                    dest=staging/'examples'/scene/f'seed_{seed}'/variant/Path(arc).name
                    ident=fp(dict(image_sha256=image_sha,xml_sha256=xml_sha,prediction=predictions[name],policy=policy,exporter_source=exporter_source,title=key))
                    old=previous.get(arc)
                    if not (old and old['render_identity']==ident and dest.is_file() and sha(dest)==old['jpeg_sha256']):
                        guard_space(root,cfg)
                        render(image,xml,predictions[name],spec['classes'],key+' | '+name,dest,policy)
                    row=dict(case=key,image=name,archive_path=arc,render_identity=ident,jpeg_sha256=sha(dest),
                        image_sha256=image_sha,annotation_sha256=xml_sha,prediction_row_sha256=fp(predictions[name]))
                    image_rows.append(row);case_images.append(name);add_file(entries,dest,arc)
                cases.append(dict(case=key,metrics_archive_path=metric_path.relative_to(root).as_posix(),metrics_sha256=expected,
                    checkpoint_sha256=result['parent_checkpoint']['sha256'],examples=case_images,count=len(case_images)))
    for path in sorted((folder/'localization').glob('*/*/*/RESULT.json')):
        value=read(path)
        if not value.get('complete'):continue
        arc=path.with_name('metrics.json').relative_to(root).as_posix()
        add_file(entries,path,arc)
    metric_count=sum(name.endswith('/metrics.json') for name in entries)
    if not metric_count:raise RuntimeError('No completed metrics for compact export; finish evaluation first')
    selection_record=dict(policy=policy,images=image_rows,cases=cases,missing_cases=missing,
        note='Same ten image identities for every configuration at each scene/seed. Image selection never uses AP, predictions or GT labels.')
    write(staging/'IMAGE_SELECTION.json',selection_record)
    arc=(folder/'IMAGE_SELECTION.json').relative_to(root).as_posix();add_file(entries,staging/'IMAGE_SELECTION.json',arc)
    manifest=dict(format=policy['format'],experiment_tag=cfg['tag'],core_complete=core['complete'],localization_complete=diag['complete'],
        metric_files=metric_count,example_images=len(image_rows),pictures_per_case=10,completed_cases=len(cases),missing_cases=missing,
        dataset_manifest_sha256=dataset_sha,exporter_source=exporter_source,
        numerical_source_lock_sha256=sha(folder/'SOURCE/LOCK.json') if (folder/'SOURCE/LOCK.json').exists() else None,
        files={name:dict(bytes=path.stat().st_size,sha256=sha(path)) for name,path in entries.items()},
        remote_raw_results_preserved=True,neural_inference_run=False,new_training_run=False,
        excluded=['weights','raw images','DINO features','full predictions','curves','calibration samples','per-image localization JSON','training plans'])
    write(staging/'COMPACT_MANIFEST.json',manifest)
    add_file(entries,staging/'COMPACT_MANIFEST.json',(folder/'COMPACT_MANIFEST.json').relative_to(root).as_posix())
    guard_space(root,cfg,sum(path.stat().st_size for path in entries.values()))
    path=folder/'REFERENCE_ABLATION_COMPACT.zip';tmp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for name,p in track(sorted(entries.items()),'159 pack metrics summaries and fixed examples',unit='files'):
            z.write(p,name,compress_type=zipfile.ZIP_STORED if p.suffix.lower()=='.jpg' else zipfile.ZIP_DEFLATED)
    with stage('159 compact ZIP CRC/SHA256'):
        with zipfile.ZipFile(tmp) as z:
            if z.testzip():raise RuntimeError('Compact ZIP CRC failed')
        digest=sha(tmp)
    tmp.replace(path)
    receipt=dict(path=str(path),bytes=path.stat().st_size,sha256=digest,crc_verified=True,metrics=metric_count,images=len(image_rows),
        core_complete=core['complete'],localization_complete=diag['complete'],weights_included=False,full_predictions_included=False)
    write(folder/'COMPACT_DOWNLOAD_RECEIPT.json',receipt)
    return receipt


def stage_upload(root,cfg):
    from refnet_ablation.protocol import stage_upload as original_stage
    result=original_stage(root,cfg);out=Path(result['upload_directory'])
    manifest=read(out/'REFERENCE_ABLATION_UPLOAD_MANIFEST.json')
    for name in ['run_reference_ablation_compact.py','src/reference_ablation_export.py','configs/reference_ablation_export.json',
                 'docs/REFERENCE_ABLATION_COMPACT.md','tests/test_reference_ablation_export.py']:
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path(root)/name,dest);manifest['files'][name]=sha(dest)
    # Include158's prerequisite code in the same upload tree. A user preparing
    #159 should not have to find and upload a separate Game training package.
    from refnet_game.protocol import stage_upload as stage_game
    game=stage_game(root,read(Path(root)/'configs/reference_game.json'))
    game_dir=Path(game['upload_directory']);game_manifest=read(game_dir/'REFERENCE_GAME_UPLOAD_MANIFEST.json')
    for name,h in game_manifest['files'].items():
        if name in manifest['files'] and manifest['files'][name]!=h:raise RuntimeError('Shared158/159 upload source differs: '+name)
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(game_dir/name,dest);manifest['files'][name]=sha(dest)
    name='REFERENCE_GAME_UPLOAD_MANIFEST.json';shutil.copy2(game_dir/name,out/name);manifest['files'][name]=sha(out/name)
    manifest['export_format']='metrics-summary-ten-images; numerical source files unchanged'
    manifest['prerequisite_game_training_included']=True
    write(out/'REFERENCE_ABLATION_UPLOAD_MANIFEST.json',manifest)
    result.update(files=len(manifest['files']),compact_export=True,game_training_included=True)
    return result


def main(root,argv=None):
    argv=list(argv or ['export']);cfg=read(Path(root)/'configs/reference_ablation.json')
    if argv[0]=='stage-upload':print(json.dumps(stage_upload(root,cfg),ensure_ascii=False,indent=2));return
    # Adapt only the final export function. Original numerical files, source
    # identities, neural code, training plans and checkpoint locks stay intact.
    from refnet_ablation import experiment
    original=experiment.export_results
    experiment.export_results=export_results
    try:experiment.main(root,argv)
    finally:experiment.export_results=original
