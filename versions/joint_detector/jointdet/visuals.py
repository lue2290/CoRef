from pathlib import Path
import torch
from PIL import Image,ImageOps,ImageDraw
from .scoring import raw_arrays
from .protocol import write_json


@torch.inference_mode()
def overfit_images(root,run,model,s,c,phase):
    root=Path(root);run=Path(run);model.eval();device=next(model.parameters()).device
    folder=run/'overfit_visual';folder.mkdir(parents=True,exist_ok=True)
    for name in s['fit_images'][:3]:
        raw=raw_arrays(model.forward_image(root,name,[],device),c)
        with Image.open(root/'assets/data/images'/name) as src:im=ImageOps.exif_transpose(src).convert('RGB')
        scale=min(1,1000/max(im.size));im=im.resize((round(im.width*scale),round(im.height*scale)))
        draw=ImageDraw.Draw(im)
        for o in s['fit_objects']:
            if o['image']==name:draw.rectangle([v*scale for v in o['box']],outline='lime',width=3)
        for b,p in zip(raw['boxes'][:15],raw['pf'][:15]):
            bb=[float(v)*scale for v in b];draw.rectangle(bb,outline='orange',width=1)
            draw.text((bb[0],bb[1]),f'{float(p):.2f}',fill='orange')
        label=f'{phase}_{Path(name).stem}'
        im.save(folder/(label+'.jpg'),quality=85)
        write_json(folder/(label+'.json'),dict(image=name,phase=phase,boxes=raw['boxes'].tolist(),pf=raw['pf'].tolist(),
                                             legend='green=selected legal GT, orange=top15 PF candidates; not final K/U detection'))
