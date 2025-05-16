import json
import tempfile
from pipeline.exporter import export_json

def test_export_json(tmp_path):
    blocks = [
        {'text':'Hi','coordinates':[0,0,10,10],'confidence':1.0,'language':'en'}
    ]
    out = tmp_path / 'out.json'
    export_json(blocks, str(out))
    data = json.loads(out.read_text(encoding='utf-8'))
    assert 'text_blocks' in data
    assert data['text_blocks'][0]['text'] == 'Hi'