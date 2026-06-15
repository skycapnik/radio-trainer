"""
Тренажёр радиообмена — Flask-версия (без Gradio, без Google).
Запуск: python app.py  →  открыть http://localhost:5000
"""

import os, re, random, copy, uuid, json, tempfile
import numpy as np
import torch
import soundfile as sf
import noisereduce as nr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_from_directory, session
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.silence import detect_silence

# ─────────────────────────── НАСТРОЙКИ ───────────────────────────
MODEL_SIZE          = 'large-v3'
READBACK_CHECK      = True
USE_NOISE_REDUCTION = True
N_SCENARIOS         = 2
MIC_DELAY_SEC       = 3

RATE_MAX_FAP                   = 100
RATE_OPTIMAL_LOW, RATE_OPTIMAL_HIGH = 60, 100
MAX_OK_PAUSE                   = 2.0
CONCISE_TOLERANCE              = 1.8
HESITATION_GAP                 = 0.7

CALLSIGNS = [
    {'digits': '1234', 'pairs': 'двенадцать тридцать четыре'},
    {'digits': '2567', 'pairs': 'двадцать пять шестьдесят семь'},
    {'digits': '7019', 'pairs': 'семьдесят ноль девятнадцать'},
    {'digits': '3041', 'pairs': 'тридцать ноль сорок один'},
    {'digits': '8806', 'pairs': 'восемьдесят восемь ноль шесть'},
]

EXTRA_OK_WORDS      = ['к','на','и','по','до','за','в','с','о','у','от','для']
ORDER_PENALTY       = 20.0
EXTRA_WORD_PENALTY  = 10.0
EXTRA_PENALTY_MAX   = 30.0

FILLER_WORDS = ['ну','это','значит','короче','типа','как бы','вот','так сказать',
                'в общем','понимаешь','понимаете','ага','угу','блин','слушай','короч']
FILLER_SOUND_PATTERNS = [r'\bэ{2,}\b', r'\bм{2,}\b', r'\bэ+м+\b', r'\bм+э+\b',
                         r'\bа{3,}\b', r'\bо{3,}\b', r'\bхм+\b', r'\bну{2,}\b']

INITIAL_PROMPT = (
    'Аэрофлот, борт, вышка, руление, разрешаю, разрешили, запуск, '
    'предварительный, исполнительный, занимаю, набираю, снижаюсь, рулю, '
    'взлёт, посадка, эшелон, курс, скорость, давление, доложу, слышу вас на пять, '
    'ВПП, РД, ветер, метров в секунду. Ээээ, ммм, эм, ну это, как бы.'
)

# ─────────────────────── БАНК СЦЕНАРИЕВ ──────────────────────────
SCENARIO_BANK = [
    {'id':1,'phase':'Проверка связи','situation':'Вы на стоянке, позывной «Аэрофлот 1234». Первый выход на связь с пунктом «Руление».',
     'controller':'«Аэрофлот 1234, Шереметьево-Руление, как меня слышите?»','data':'Позывной: Аэрофлот 1234 («двенадцать тридцать четыре»). Слышимость отличная.',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['слышу вас на пять','слышу на пять','слышу вас хорошо','слышу вас отлично','на пять'],'crit':True,'label':'оценка слышимости'}],
     'numbers':[],'model_answer':'«Слышу вас на пять, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Оцените слышимость по 5-балльной шкале и назовите позывной.','antagonists':[]},
    {'id':2,'phase':'Запуск двигателей','situation':'Вы на перроне, запросили запуск двигателей.',
     'controller':'«Аэрофлот 1234, запуск разрешаю».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['запуск разрешили','запуск разрешаю','запуск произвожу','произвожу запуск'],'crit':True,'label':'подтверждение запуска'}],
     'numbers':[],'model_answer':'«Запуск разрешили, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Повторите разрешение и назовите позывной. Кратко, без лишних слов.',
     'antagonists':['посадк','взлет','взлёт','снижа']},
    {'id':3,'phase':'Руление','situation':'Двигатели запущены, вы готовы к рулению. Аэродром Шереметьево.',
     'controller':'«Аэрофлот 1234, рулите к предварительному ВПП 24 по РД5 и М».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['рулю','рулим'],'crit':False,'label':'глагол «рулю»'},
                 {'alts':['предварительн'],'crit':False,'label':'«к предварительному»'},
                 {'alts':['двадцать четыре','24'],'crit':True,'label':'номер ВПП 24'},
                 {'alts':['рд5','рд 5','по рд','м','по м'],'crit':False,'label':'маршрут руления (РД5, М)'}],
     'numbers':['24'],'model_answer':'«Рулю к предварительному ВПП двадцать четыре по РД5 и М, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Маршрут руления и номер ВПП повторяются обязательно — это требование квитирования.',
     'antagonists':['взлет','взлёт','посадк']},
    {'id':4,'phase':'Исполнительный старт','situation':'Вы на предварительном старте ВПП 24, готовы занять исполнительный.',
     'controller':'«Аэрофлот 1234, занимайте исполнительный ВПП 24».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['занимаю исполнительный','исполнительный занимаю','занимаем исполнительный'],'crit':True,'label':'занятие исполнительного'},
                 {'alts':['двадцать четыре','24'],'crit':True,'label':'номер ВПП 24'}],
     'numbers':['24'],'model_answer':'«Занимаю исполнительный ВПП двадцать четыре, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Подтвердите занятие исполнительного и обязательно повторите номер ВПП.',
     'antagonists':['посадк','запуск']},
    {'id':5,'phase':'Взлёт','situation':'Вы на исполнительном старте ВПП 24, готовы к взлёту.',
     'controller':'«Аэрофлот 1234, ветер 240 градусов 5 метров в секунду, ВПП 24, взлёт разрешаю».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['взлёт разрешили','взлет разрешили','взлёт разрешаю','разрешили взлёт','выполняю взлёт'],'crit':True,'label':'подтверждение взлёта'}],
     'numbers':[],'model_answer':'«Взлёт разрешили, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Ветер — справочная информация, НЕ повторяется. Повторите только разрешение и позывной.',
     'antagonists':['посадк','снижа']},
    {'id':6,'phase':'Набор эшелона','level_kind':('эшелон','350'),'situation':'Вы после взлёта, диспетчер даёт команду набора.',
     'controller':'«Аэрофлот 1234, набирайте эшелон 350».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['набираю','набираем'],'crit':False,'label':'глагол «набираю»'},
                 {'alts':['эшелон','эшелона'],'crit':True,'label':'слово «эшелон»'},
                 {'alts':['триста пятьдесят','три пять ноль','350'],'crit':True,'label':'значение эшелона 350'}],
     'numbers':['350'],'model_answer':'«Набираю эшелон триста пятьдесят, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Эшелон — критический параметр, повторяется обязательно и точно.',
     'antagonists':['снижа','посадк']},
    {'id':7,'phase':'Изменение курса','level_kind':('курс','270'),'situation':'Вы в наборе, диспетчер задаёт курс для эшелонирования.',
     'controller':'«Аэрофлот 1234, для эшелонирования курс 270».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['курс','курсом'],'crit':True,'label':'слово «курс»'},
                 {'alts':['двести семьдесят','два семь ноль','270'],'crit':True,'label':'значение курса 270'}],
     'numbers':['270'],'model_answer':'«Курс двести семьдесят, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Курс обязателен к повторению. Причину («для эшелонирования») повторять не нужно.',
     'antagonists':['посадк','запуск']},
    {'id':8,'phase':'Снижение','level_kind':('эшелон','110'),'situation':'Вы в крейсерском полёте на эшелоне 350, начинаете снижение.',
     'controller':'«Аэрофлот 1234, снижайтесь эшелон 110».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['снижаюсь','снижаемся'],'crit':False,'label':'глагол «снижаюсь»'},
                 {'alts':['эшелон','эшелона'],'crit':True,'label':'слово «эшелон»'},
                 {'alts':['сто десять','один один ноль','110'],'crit':True,'label':'значение эшелона 110'}],
     'numbers':['110'],'model_answer':'«Снижаюсь эшелон сто десять, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Новый эшелон повторяется обязательно.',
     'antagonists':['набира','набор','взлет','взлёт']},
    {'id':9,'phase':'Установка давления','digit_readout':True,'situation':'Вы подходите к району аэродрома, диспетчер передаёт давление.',
     'controller':'«Аэрофлот 1234, давление 1013».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['давление'],'crit':True,'label':'слово «давление»'},
                 {'alts':['один ноль один три','десять тринадцать','1013'],'crit':True,'label':'значение давления 1013'}],
     'numbers':['1013'],'model_answer':'«Давление один ноль один три, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Установка давления (QNH) обязательна к квитированию.',
     'antagonists':['взлет','взлёт','посадк']},
    {'id':10,'phase':'Доклад скорости','situation':'Вы в полёте, диспетчер запрашивает приборную скорость.',
     'controller':'«Аэрофлот 1234, доложите приборную скорость».','data':'Позывной: Аэрофлот 1234 («двенадцать тридцать четыре»). Приборная скорость: 480 км/ч.',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['скорость'],'crit':True,'label':'слово «скорость»'},
                 {'alts':['четыреста восемьдесят','четыре восемь ноль','480'],'crit':True,'label':'значение скорости 480'}],
     'numbers':['480'],'model_answer':'«Скорость четыреста восемьдесят, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Это доклад своих данных. Назовите параметр и точное значение.',
     'antagonists':[]},
    {'id':11,'phase':'Посадка','situation':'Вы на предпосадочной прямой ВПП 24, готовы к посадке.',
     'controller':'«Аэрофлот 1234, ветер 230 градусов 4 метра в секунду, ВПП 24, посадку разрешаю».','data':'Позывной: Аэрофлот 1234 — произносится «двенадцать тридцать четыре»',
     'elements':[{'alts':['аэрофлот двенадцать тридцать четыре','двенадцать тридцать четыре','1234'],'crit':True,'callsign':True,'label':'позывной'},
                 {'alts':['посадку разрешили','посадка разрешена','разрешили посадку','выполняю посадку'],'crit':True,'label':'подтверждение посадки'}],
     'numbers':[],'model_answer':'«Посадку разрешили, Аэрофлот двенадцать тридцать четыре».',
     'tip':'Ветер не повторяется. Повторите только разрешение на посадку и позывной.',
     'antagonists':['взлет','взлёт']},
]
for s in SCENARIO_BANK:
    s['model_words'] = len([w for w in re.sub(r'[^\w\s]',' ',s['model_answer']).split() if w])

# ─────────────────────── ЧИСЛИТЕЛЬНЫЕ ────────────────────────────
_UNITS = {'ноль':0,'нуль':0,'один':1,'одна':1,'два':2,'две':2,'три':3,'четыре':4,
          'пять':5,'шесть':6,'семь':7,'восемь':8,'девять':9}
_TEENS = {'десять':10,'одиннадцать':11,'двенадцать':12,'тринадцать':13,'четырнадцать':14,
          'пятнадцать':15,'шестнадцать':16,'семнадцать':17,'восемнадцать':18,'девятнадцать':19}
_TENS  = {'двадцать':20,'тридцать':30,'сорок':40,'пятьдесят':50,'шестьдесят':60,
          'семьдесят':70,'восемьдесят':80,'девяносто':90}
_HUND  = {'сто':100,'двести':200,'триста':300,'четыреста':400,'пятьсот':500,
          'шестьсот':600,'семьсот':700,'восемьсот':800,'девятьсот':900}
_ALLNUM = set(_UNITS)|set(_TEENS)|set(_TENS)|set(_HUND)

def _wcls(w):
    if w in _HUND: return 'H'
    if w in _TENS:  return 'T'
    if w in _TEENS: return 'E'
    if w in _UNITS: return 'U'
    return None

def _words_to_digits(toks):
    res=''; i=0; n=len(toks)
    while i<n:
        w=toks[i]; c=_wcls(w)
        if c is None: i+=1; continue
        if c=='H':
            v=_HUND[w]; i+=1
            if i<n and _wcls(toks[i])=='T':
                v+=_TENS[toks[i]]; i+=1
                if i<n and _wcls(toks[i])=='U': v+=_UNITS[toks[i]]; i+=1
            elif i<n and _wcls(toks[i])=='E': v+=_TEENS[toks[i]]; i+=1
            elif i<n and _wcls(toks[i])=='U': v+=_UNITS[toks[i]]; i+=1
            res+=str(v)
        elif c=='T':
            v=_TENS[w]; i+=1
            if i<n and _wcls(toks[i])=='U': v+=_UNITS[toks[i]]; i+=1
            res+=str(v)
        elif c=='E': res+=str(_TEENS[w]); i+=1
        elif c=='U': res+=str(_UNITS[w]); i+=1
    return res

def canon_numbers(text):
    toks=normalize(text).split(); out=[]; i=0; n=len(toks)
    while i<n:
        if toks[i] in _ALLNUM:
            j=i; grp=[]
            while j<n and toks[j] in _ALLNUM: grp.append(toks[j]); j+=1
            out.append(_words_to_digits(grp)); i=j
        else:
            out.append(toks[i]); i+=1
    return ' '.join(out)

# ─────────────────────────── УТИЛИТЫ ─────────────────────────────
def normalize(t):
    return re.sub(r'[^\w\s]',' ',t.lower()).strip()

def apply_callsign(scn, cs):
    s=copy.deepcopy(scn); d,p=cs['digits'],cs['pairs']
    for key in ('controller','data','model_answer','situation'):
        if key in s:
            s[key]=s[key].replace('Аэрофлот 1234',f'Аэрофлот {d}').replace('двенадцать тридцать четыре',p)
    for el in s['elements']:
        if el.get('callsign'):
            el['alts']=[f'аэрофлот {p}',p,d]
    s['callsign_digits']=d
    s['model_words']=len([w for w in re.sub(r'[^\w\s]',' ',s['model_answer']).split() if w])
    return s

def callsign_present(text, callsign_number='1234'):
    canon=canon_numbers(text)
    return bool(re.search(r'\b'+re.escape(callsign_number)+r'\b', canon))

def callsign_pronunciation_warn(text):
    raw=normalize(text)
    if re.search(r'\bодин\s+два\s+три\s+четыре\b',raw) or re.search(r'\b1\s+2\s+3\s+4\b',raw):
        return ('ℹ️ Похоже, позывной произнесён по одной цифре. По правилам '
                'четырёхзначный позывной произносится парами: «двенадцать тридцать четыре». '
                '(Это подсказка; на оценку не влияет.)')
    return None

def preprocess_audio(in_path, out_path):
    a=AudioSegment.from_file(in_path).set_channels(1).set_frame_rate(16000).normalize()
    tmp=out_path+'.tmp.wav'; a.export(tmp, format='wav')
    if USE_NOISE_REDUCTION:
        d,sr=sf.read(tmp)
        sf.write(out_path, nr.reduce_noise(y=d, sr=sr, stationary=True, prop_decrease=0.8), sr)
    else:
        a.export(out_path, format='wav')
    if os.path.exists(tmp): os.remove(tmp)
    return out_path

def transcribe(path, scenario_prompt=''):
    segs,info=model.transcribe(
        path, language='ru', initial_prompt=INITIAL_PROMPT+' '+scenario_prompt,
        vad_filter=True, beam_size=5, temperature=0.0,
        condition_on_previous_text=False, suppress_tokens=[], word_timestamps=True)
    segs=list(segs)
    text=' '.join(s.text.strip() for s in segs).strip()
    lp=[s.avg_logprob for s in segs]
    words=[]
    for s in segs:
        if getattr(s,'words',None):
            for w in s.words: words.append((w.start,w.end))
    return text, (float(np.mean(lp)) if lp else None), words

def hesitation_gaps(words):
    g=0
    for i in range(1,len(words)):
        if words[i][0]-words[i-1][1]>=HESITATION_GAP: g+=1
    return g

def detect_filled_pauses_audio(path, min_dur=0.35, hop=0.02):
    try:
        y,sr=sf.read(path)
        if y.ndim>1: y=y.mean(axis=1)
        y=y.astype(np.float64)
    except Exception:
        return 0,0.0
    win=int(0.04*sr); step=int(hop*sr)
    if len(y)<win: return 0,0.0
    energies,periodicity,zcr=[],[],[]
    for i in range(0,len(y)-win,step):
        fr=y[i:i+win]; energies.append(np.sqrt(np.mean(fr**2)))
        zcr.append(np.mean(np.abs(np.diff(np.sign(fr))))/2)
        fr=fr-fr.mean(); ac=np.correlate(fr,fr,'full')[len(fr)-1:]
        if ac[0]<=1e-9: periodicity.append(0.0); continue
        ac=ac/ac[0]; lo,hi=int(sr/320),int(sr/80)
        periodicity.append(float(np.max(ac[lo:min(hi,len(ac))])))
    energies=np.array(energies); periodicity=np.array(periodicity); zcr=np.array(zcr)
    if len(energies)==0: return 0,0.0
    e_thr=max(energies.max()*0.15,1e-4)
    voiced=(energies>e_thr)&(periodicity>0.6)&(zcr<0.12)
    count,total=0,0.0; min_frames=int(min_dur/hop); i=0; n=len(voiced)
    while i<n:
        if voiced[i]:
            j=i
            while j<n and voiced[j]: j+=1
            seg_len=j-i
            if seg_len>=min_frames:
                seg_e=energies[i:j]; seg_p=periodicity[i:j]
                cv_e=np.std(seg_e)/(np.mean(seg_e)+1e-9)
                if cv_e<0.5 and np.mean(seg_p)>0.6: count+=1; total+=seg_len*hop
            i=j
        else: i+=1
    return count,round(total,2)

def detect_fillers(text):
    raw=text.lower(); collapsed=_collapse_repeats(raw); counts={}
    for pat in FILLER_SOUND_PATTERNS:
        for m in re.findall(pat,collapsed): counts[m]=counts.get(m,0)+1
    norm=re.sub(r'[^\w\s\-]',' ',raw)
    for f in FILLER_WORDS:
        n=len(re.findall(r'\b'+re.escape(f)+r'\b',norm))
        if n: counts[f]=counts.get(f,0)+n
    return counts,sum(counts.values())

def _collapse_repeats(t):
    for ch in 'эмао':
        t=re.sub(r'\b'+ch+r'(?:[\s\-]*'+ch+r')+\b',lambda m,c=ch:c*m.group().count(c),t)
    return t

def analyze_pauses(path, min_ms=500, thr=-40):
    a=AudioSegment.from_file(path)
    sil=detect_silence(a,min_silence_len=min_ms,silence_thresh=thr)
    return sil,[(e-s)/1000.0 for s,e in sil]

def volume_consistency(path):
    try:
        y,sr=sf.read(path)
        if y.ndim>1: y=y.mean(axis=1)
        y=y.astype(np.float64)
    except Exception:
        return 80.0
    win=int(0.04*sr); step=int(0.03*sr)
    if len(y)<win: return 80.0
    dbs=[]
    for i in range(0,len(y)-win,step):
        rms=np.sqrt(np.mean(y[i:i+win]**2))
        if rms>1e-6: dbs.append(20*np.log10(rms))
    if not dbs: return 80.0
    return max(0.0, min(100.0, 100.0 - max(0.0, np.std(dbs)-6)*3))

def detect_prolonged_vowels(path, hop=0.02, max_normal=0.30):
    try:
        y,sr=sf.read(path)
        if y.ndim>1: y=y.mean(axis=1)
        y=y.astype(np.float64)
    except Exception:
        return 0,0.0
    win=int(0.04*sr); step=int(hop*sr)
    if len(y)<win: return 0,0.0
    energies,periodicity=[],[]
    for i in range(0,len(y)-win,step):
        fr=y[i:i+win]; energies.append(np.sqrt(np.mean(fr**2)))
        fr2=fr-fr.mean(); ac=np.correlate(fr2,fr2,'full')[len(fr2)-1:]
        if ac[0]<=1e-9: periodicity.append(0.0); continue
        ac=ac/ac[0]; lo,hi=int(sr/320),int(sr/80)
        periodicity.append(float(np.max(ac[lo:min(hi,len(ac))])))
    energies=np.array(energies); periodicity=np.array(periodicity)
    if len(energies)==0: return 0,0.0
    e_thr=max(energies.max()*0.15,1e-4)
    voiced=(energies>e_thr)&(periodicity>0.6)
    count=0; extra=0.0; min_frames=int(max_normal/hop)
    i=0; n=len(voiced)
    while i<n:
        if voiced[i]:
            j=i
            while j<n and voiced[j]: j+=1
            dur=(j-i)*hop
            if dur>max_normal: count+=1; extra+=dur-max_normal
            i=j
        else: i+=1
    return count,round(extra,2)

def match_numbers(text, numbers):
    if not numbers: return True,[],[]
    found,miss=[],[]
    for num in numbers:
        canon=canon_numbers(text)
        if re.search(r'\b'+re.escape(num)+r'\b',canon): found.append(num)
        else: miss.append(num)
    return not miss,found,miss

def match_elements(text, elements):
    norm=normalize(text); matched=[]; missing=[]
    total=len(elements); found=0
    for el in elements:
        hit=any(a in norm for a in el['alts'])
        if hit: found+=1; matched.append(el['label'])
        elif el['crit']: missing.append(el['label'])
    pct=(found/total*100) if total>0 else 100.0
    rb_pct=(found/total*100) if total>0 else 100.0
    return pct,rb_pct,matched,missing

def _find_positions(text, elements):
    norm=normalize(text); positions={}
    for i,el in enumerate(elements):
        for alt in el['alts']:
            idx=norm.find(alt)
            if idx>=0: positions[i]=idx; break
    return positions

def check_order(text, elements):
    pos=_find_positions(text,elements)
    if len(pos)<2: return True,len(pos)
    vals=[pos[k] for k in sorted(pos.keys())]
    return vals==sorted(vals),len(pos)

def extra_words(text, elements, extra_ok, model_answer=''):
    norm=normalize(text); model_norm=normalize(model_answer)
    expected=set()
    for el in elements:
        for a in el['alts']: expected.update(normalize(a).split())
    expected.update(normalize(model_norm).split()); expected.update(extra_ok)
    toks=[w for w in norm.split() if w and len(w)>1]
    extras=[w for w in toks if w not in expected and w not in extra_ok]
    return extras

def score_fluency_only(pr, nf, gaps):
    s=100.0
    if pr>0.25: s-=(pr-0.25)*60
    s-=nf*8; s-=gaps*5
    return max(0.0,min(100.0,s))

def clarity_from_logprob(lp):
    if lp is None: return 3.0
    return max(1.0,min(5.0,1.0+(lp+0.8)/0.16))

def fap_intelligibility(c):
    if c>=4.5: return 5,'отлично'
    if c>=3.5: return 4,'хорошо'
    if c>=2.5: return 3,'удовлетворительно'
    if c>=1.5: return 2,'плохо'
    return 1,'неразборчиво'

def score_conciseness(rw, mw):
    if mw==0: return 100.0
    ratio=rw/mw
    if ratio<=1.0: return 100.0
    if ratio<=CONCISE_TOLERANCE: return 100.0-(ratio-1.0)*40
    return max(0.0,100.0-(ratio-1.0)*80)

def detect_antagonist(text, antagonists):
    if not antagonists: return []
    norm=normalize(text)
    return [a for a in antagonists if re.search(r'\b'+re.escape(a),norm)]

def detect_negation(text):
    norm=normalize(text)
    return [m.group() for m in re.finditer(r'\bне\s+\w+',norm)]

def leading_silence_penalty(first_word_start):
    if first_word_start is None or first_word_start<=0.5: return 0.0
    if first_word_start<=1.0: return (first_word_start-0.5)*20
    return 10.0+min(20.0,(first_word_start-1.0)*15)

def check_pressure_pronunciation(text):
    norm=normalize(text)
    digit_pat=r'\b(один|одна|ноль|два|три|четыре|пять|шесть|семь|восемь|девять)\b'
    if re.search(r'\b(тысяча\s+(ноль\s+)?тринадцать|тысяча\s+тринадцать)\b',norm):
        return 'wrong','Сказано «тысяча тринадцать» — нужно поцифрово: «один ноль один три»'
    digits=re.findall(digit_pat,norm)
    if len(digits)>=3: return 'ok','Давление прочитано поцифрово ✅'
    return 'unknown','Не удалось определить способ произношения давления'

def check_zero_rule(text, kind, value):
    norm=normalize(text)
    if kind=='эшелон':
        if '110' in value or 'сто десять' in norm: return 'ok','Ноль в начале эшелона не требуется — всё верно'
        if re.search(r'\bноль\b',norm): return 'ok','Ноль произнесён — соответствует п.11 ФАП-414'
    elif kind=='курс':
        if re.search(r'\bноль\b',norm): return 'ok','Ноль в курсе произнесён'
    return 'na',''

def critical_violations(callsign_ok, numbers, n_miss, antag_hits, negations=None):
    v=[]
    if not callsign_ok: v.append('Позывной не назван (п.25, 33, 53 ФАП-414) — автоматический незачёт')
    if n_miss: v.append(f'Не повторены числовые значения: {", ".join(n_miss)} (п.50 ФАП-414)')
    if antag_hits: v.append(f'Ответ не соответствует команде диспетчера: упомянуто «{"", "".join(antag_hits)}»')
    if negations: v.append(f'Обнаружено отрицание в ответе: {", ".join(negations)}')
    return v

def quality_criteria(f):
    diction=min(100.0, f.get('clarity',3)/5*100 * (f.get('volume_consistency',80)/100)**0.5)
    fluency=score_fluency_only(f.get('pause_ratio',0), f.get('n_fillers',0), f.get('gaps',0))
    struct=f.get('element_pct',0)
    rb=f.get('readback_completeness',100)
    return {
        'Дикция':          round(diction,1),
        'Чистота речи':    round(fluency,1),
        'Структура передачи': round(struct,1),
        'Квитирование':    round(rb,1),
        'Скорость речи':   round(min(100.0,f.get('wpm',80)/100*100),1),
    }

def grade(headline, weakest, critical=None):
    if critical: return 'Неудовлетворительно','❌'
    if headline>=85 and weakest>=70: return 'Отлично','🏆'
    if headline>=70 and weakest>=55: return 'Хорошо','✅'
    if headline>=55 and weakest>=40: return 'Удовлетворительно','⚠️'
    return 'Неудовлетворительно','❌'

def make_chart(crit, save_dir, idx):
    keys=[k for k in crit if k!='Скорость речи']
    vals=[crit[k] for k in keys]
    colors=['#e74c3c' if v<55 else '#f39c12' if v<75 else '#27ae60' for v in vals]
    fig,ax=plt.subplots(figsize=(6,3.5))
    bars=ax.barh(keys,vals,color=colors,height=0.55,edgecolor='none')
    ax.set_xlim(0,105); ax.axvline(55,color='#e74c3c',lw=1,ls='--',alpha=0.5)
    ax.axvline(75,color='#27ae60',lw=1,ls='--',alpha=0.5)
    ax.set_xlabel('Баллы',fontsize=9); ax.tick_params(labelsize=9)
    for b,v in zip(bars,vals):
        ax.text(b.get_width()+1,b.get_y()+b.get_height()/2,f'{v:.0f}',va='center',fontsize=9)
    ax.set_title('Показатели качества (ФАП-414)',fontsize=10,pad=8)
    fig.tight_layout()
    fname=f'chart_{idx}.png'; fpath=os.path.join(save_dir,fname)
    fig.savefig(fpath,dpi=100,bbox_inches='tight'); plt.close(fig)
    return fname

# ──────────────────────── FLASK APP ──────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

SESSIONS = {}   # session_id -> state
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)
AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'static', 'audio_tmp')
os.makedirs(AUDIO_DIR, exist_ok=True)

print('⏳ Загружаю модель Whisper...')
DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
COMPUTE = 'float16' if DEVICE=='cuda' else 'int8'
model   = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
print(f'✅ Модель загружена ({DEVICE})')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/start', methods=['POST'])
def api_start():
    sid = str(uuid.uuid4())
    base = random.sample(SCENARIO_BANK, N_SCENARIOS)
    sel  = [apply_callsign(s, random.choice(CALLSIGNS)) for s in base]
    SESSIONS[sid] = {'sel': sel, 'cur': 0, 'res': []}
    s = sel[0]
    return jsonify({'session_id': sid, 'total': N_SCENARIOS,
                    'scenario_num': 1, 'phase': s['phase'], 'situation': s['situation']})

@app.route('/api/ready', methods=['POST'])
def api_ready():
    data = request.json
    st = SESSIONS.get(data.get('session_id'))
    if not st: return jsonify({'error': 'session not found'}), 400
    s = st['sel'][st['cur']]
    return jsonify({'controller': s['controller'], 'data': s['data'],
                    'delay': MIC_DELAY_SEC})

@app.route('/api/submit', methods=['POST'])
def api_submit():
    sid = request.form.get('session_id')
    st  = SESSIONS.get(sid)
    if not st: return jsonify({'error': 'session not found'}), 400
    if 'audio' not in request.files:
        return jsonify({'error': 'no audio'}), 400

    audio_file = request.files['audio']
    raw_path   = os.path.join(AUDIO_DIR, f'{sid}_{st["cur"]}_raw')
    clean_path = os.path.join(AUDIO_DIR, f'{sid}_{st["cur"]}_clean.wav')
    audio_file.save(raw_path)

    try:
        clean = preprocess_audio(raw_path, clean_path)
        idx   = st['cur']
        s     = st['sel'][idx]
        text, lp, words = transcribe(clean, s['controller'])
        dur = len(AudioSegment.from_file(clean))/1000.0
        _,pauses = analyze_pauses(clean)
        pr = sum(pauses)/dur if dur>0 else 0
        toks=[w for w in text.split() if w]; nwords=len(toks)
        wpm = nwords/(max(0.1,dur-sum(pauses))/60)
        fl_counts,nf = detect_fillers(text)
        gaps = hesitation_gaps(words)
        ac_count,ac_dur = detect_filled_pauses_audio(clean)
        pv_count,pv_extra = detect_prolonged_vowels(clean)
        nf_total = nf+gaps+ac_count+pv_count
        el_pct_base,rb,matched,missing = match_elements(text,s['elements'])
        order_ok,n_content = check_order(text,s['elements'])
        order_pen = ORDER_PENALTY if (n_content>=2 and not order_ok) else 0.0
        extras = extra_words(text,s['elements'],EXTRA_OK_WORDS,s['model_answer'])
        extra_pen = min(EXTRA_PENALTY_MAX,len(extras)*EXTRA_WORD_PENALTY)
        el_pct = max(0.0,el_pct_base-order_pen-extra_pen)
        nm,n_found,n_miss = match_numbers(text,s['numbers'])
        clarity = clarity_from_logprob(lp)
        vol = volume_consistency(clean)
        concise = score_conciseness(nwords,s['model_words'])
        pairing_note = callsign_pronunciation_warn(text)
        cs_digits = s.get('callsign_digits','1234')
        callsign_ok = callsign_present(text,cs_digits)
        antag_hits  = detect_antagonist(text,s.get('antagonists',[]))
        negations   = detect_negation(text)
        first_word_start = words[0][0] if words else None
        lead_pen = leading_silence_penalty(first_word_start)

        feat={'clarity':clarity,'volume_consistency':vol,'wpm':wpm,'pause_ratio':pr,
              'n_fillers':nf_total,'numbers_match':nm,'callsign':callsign_ok,
              'readback_completeness':rb,'element_pct':el_pct,'conciseness':concise,
              'resp_words':nwords,'gaps':gaps}
        crit_scores = quality_criteria(feat)
        crit_scores['Чистота речи'] = round(max(0.0,crit_scores['Чистота речи']-lead_pen),1)
        weakest = min(v for k,v in crit_scores.items() if k!='Скорость речи')

        pressure_status,pressure_msg=('na','')
        if s.get('digit_readout'):
            pressure_status,pressure_msg=check_pressure_pronunciation(text)
        pressure_critical=['Давление прочитано неверно'] if pressure_status=='wrong' else []

        zero_status,zero_msg=('na','')
        if s.get('level_kind'):
            zk,zv=s['level_kind']
            zero_status,zero_msg=check_zero_rule(text,zk,zv)

        critical = critical_violations(callsign_ok,s['numbers'],n_miss,antag_hits,negations)+pressure_critical
        sq=(crit_scores['Дикция']+crit_scores['Чистота речи'])/2
        headline = round(0.7*el_pct+0.3*sq,1)
        g_name,g_mk = grade(headline,weakest,critical)
        intel,intel_txt = fap_intelligibility(clarity)
        weak_name = min(crit_scores,key=crit_scores.get)
        key_cmd_ok=(len(antag_hits)==0)and(len(negations)==0)and(el_pct_base>=50)

        st['res'].append({'phase':s['phase'],'headline':headline,'crit':crit_scores,
                          'grade':g_name,'critical':critical})

        chart_file = make_chart(crit_scores, CHARTS_DIR, f'{sid}_{idx}')

        hints=[]
        if critical:
            for c in critical: hints.append({'type':'critical','text':c})
        if matched: hints.append({'type':'ok','text':'Присутствуют: '+', '.join(matched)})
        if missing:  hints.append({'type':'err','text':'Пропущено: '+', '.join(missing)})
        if order_pen>0: hints.append({'type':'warn','text':f'Нарушен порядок ключевых слов (−{order_pen:.0f})'})
        if extras: hints.append({'type':'warn','text':f'Лишние слова: {", ".join(extras)} (−{extra_pen:.0f})'})
        if pairing_note: hints.append({'type':'info','text':pairing_note})
        if nf_total>0:
            parts=[]
            if fl_counts: parts.append('слова-паразиты: '+', '.join(f'«{k}» ×{v}' for k,v in fl_counts.items()))
            if ac_count: parts.append(f'затянутых «эээ/ммм»: {ac_count} (~{ac_dur}с)')
            if pv_count: parts.append(f'растянутых гласных: {pv_count} (~{pv_extra}с)')
            if gaps: parts.append(f'колебательных пауз: {gaps}')
            hints.append({'type':'warn','text':'Паузы-паразиты: '+'; '.join(parts)})
        if lead_pen>0 and first_word_start:
            hints.append({'type':'warn','text':f'Долгая начальная пауза ({first_word_start:.1f}с до первого слова)'})
        if s.get('digit_readout') and pressure_msg:
            hints.append({'type':'info' if pressure_status!='wrong' else 'err','text':'Давление: '+pressure_msg})
        if zero_msg:
            hints.append({'type':'info','text':'Правило нуля (п.11): '+zero_msg})

        metrics=[
            {'label':'Скорость речи (информационно)','value':f'{wpm:.0f} сл/мин','status':'info'},
            {'label':'Разборчивость (5-балльная)','value':f'{intel} — {intel_txt}','status':'ok' if intel>=4 else 'warn'},
            {'label':'Фразеология (п.92–94)','value':f'{el_pct:.0f}%','status':'ok' if el_pct>=75 else 'warn'},
            {'label':'Квитирование чисел (п.50)','value':'все' if not n_miss else f'нет: {", ".join(n_miss)}','status':'ok' if not n_miss else 'err'},
            {'label':'Позывной (п.25,33,53)','value':'есть' if callsign_ok else 'нет','status':'ok' if callsign_ok else 'err'},
            {'label':'Паузы-паразиты','value':str(nf_total),'status':'ok' if nf_total==0 else 'err'},
            {'label':'Соответствие команде','value':'да' if key_cmd_ok else 'нет','status':'ok' if key_cmd_ok else 'err'},
            {'label':'Чистота речи','value':f'{crit_scores["Чистота речи"]:.0f}/100','status':'ok' if crit_scores["Чистота речи"]>=55 else 'warn'},
            {'label':'Громкость (п.8б)','value':f'{vol:.0f}/100','status':'ok' if vol>=60 else 'warn'},
        ]

        last = (idx == len(st['sel'])-1)
        return jsonify({
            'transcribed': text,
            'model_answer': s['model_answer'],
            'tip': s['tip'],
            'headline': headline,
            'grade': g_name,
            'grade_icon': g_mk,
            'weak_name': weak_name,
            'weak_val': crit_scores[weak_name],
            'key_cmd_ok': key_cmd_ok,
            'hints': hints,
            'metrics': metrics,
            'chart_url': f'/static/charts/{chart_file}',
            'is_last': last,
            'wpm': round(wpm,1),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        for p in [raw_path]:
            if os.path.exists(p): os.remove(p)

@app.route('/api/next', methods=['POST'])
def api_next():
    data = request.json
    st   = SESSIONS.get(data.get('session_id'))
    if not st: return jsonify({'error': 'session not found'}), 400
    st['cur'] += 1
    if st['cur'] < len(st['sel']):
        s = st['sel'][st['cur']]
        return jsonify({'done': False, 'scenario_num': st['cur']+1,
                        'total': N_SCENARIOS, 'phase': s['phase'], 'situation': s['situation']})
    # Итоги
    res = st['res']; scores=[r['headline'] for r in res]
    final = float(np.mean(scores)) if scores else 0
    weakest_all = min(min(v for k,v in r['crit'].items() if k!='Скорость речи') for r in res) if res else 0
    any_critical = any(r.get('critical') for r in res)
    g_name,g_mk = grade(final,weakest_all,['x'] if any_critical else None)
    verdict = 'ЗАЧЁТ' if g_name!='Неудовлетворительно' else 'НЕЗАЧЁТ'
    items=[{'phase':r['phase'],'score':r['headline'],'grade':r['grade'],'has_critical':bool(r.get('critical'))} for r in res]
    return jsonify({'done': True, 'final': round(final,1), 'grade': g_name, 'grade_icon': g_mk,
                    'verdict': verdict, 'any_critical': any_critical, 'items': items})

if __name__ == '__main__':
    print('\n✈️  Тренажёр радиообмена запущен')
    print('   Откройте браузер: http://localhost:5000\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
