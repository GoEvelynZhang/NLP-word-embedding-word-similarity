from argparse import ArgumentParser
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
nltk.download('wordnet')

parser = ArgumentParser()

parser.add_argument("-e", "--embedding", dest = "emb_path",
    required = True, help = "path to your embedding")

args = parser.parse_args()

words = {'acronymic', 'implicational', 'shouter', 'fractures', 'endurable',
'season', 'interplanetary', 'panic', 'fastness', 'disinvestment', 'up-to-date',
'admiralty', 'murder', 'loss', 'rejoinders', 'cash', 'metal', 'exhibit',
'exterminate', 'disinheritance', 'churchs', 'discriminate', 'vulgarism',
'recourse', 'deciphering', 'partible', 'marriage', 'meet', 'houseful',
'unemotional', 'nest', 'sodium', 'carnivore', 'circumscribes', 'listeners',
'continuance', 'cylindrical', 'undeniable', 'preschoolers', 'courteous',
'establishment', 'submerging', 'grow', 'improvised', 'shrink', 'sandbag',
'prudent', 'sexually', 'france', 'moralist', 'affiliation', 'householders',
'evidently', 'provisionally', 'sun', 'monotony', 'trusteeship', 'conversely',
'indirectness', 'transverse', 'seafood', 'incensing', 'immigrating', 'rainless',
'harpsichord', 'rational', 'intentionality', 'hypocrisy', 'common',
'distribute', 'undefined', 'combust', 'religionist', 'occlusion', 'recognition',
'medal', 'partiality', 'wrong', 'paradoxical', 'benchmark', 'episode',
'unbroken', 'brightness', 'unconsolidated', 'payment', 'unmentionables', 'skip',
'school', 'motherless', 'soldier', 'tell', 'seal', 'hole', 'sheikhdoms',
'basic', 'peripheral', 'shower', 'omnipotence', 'retrying', 'boisterously',
'detestable', 'defecation', 'child', 'abandon', 'inversions', 'leagued',
'pretenders', 'soap', 'chance', 'mile', 'deny', 'capitalised', 'terror',
'inventively', 'finance', 'evacuated', 'prostatic', 'bend', 'worthless',
'nation', 'variable', 'refer', 'rank', 'doctor', 'death', 'fret', 'policy',
'back', 'furnace', 'importances', 'affectional', 'canyon', 'bronchus',
'possible', 'civilise', 'interpenetrate', 'villainous', 'production', 'market',
'comportment', 'conjecture', 'algebraist', 'contortionists', 'translocating',
'entrapping', 'defrauding', 'brave', 'artist', 'smile', 'shoulders',
'attributions', 'legalism', 'pervert', 'regiment', 'possibility', 'bachelors',
'citizen', 'decapitated', 'conflict', 'cold', 'return', 'naughtiness',
'careful', 'month', 'hypertext', 'cock', 'hyperextension', 'transforming',
'glittery', 'text', 'procreation', 'pessimist', 'typify', 'alcohol',
'disturbing', 'marginality', 'noticeable', 'preventive', 'charm', 'libelous',
'homosexual', 'bread', 'covariant', 'measurements', 'congeniality',
'self-fulfillment', 'take', 'initiation', 'ranch', 'deposes', 'harmony',
'drafting', 'butter', 'feel', 'instrumentality', 'ingroup', 'heterosexual',
'curious', 'recess', 'anxiety', 'make', 'assembly', 'distrustful', 'asteroidal',
'leading', 'copy', 'corroding', 'solemnity', 'pinpointed', 'manacles',
'brigadier', 'tasteful', 'utility', 'responsible', 'containerful',
'conscientiousness', 'circumcisions', 'unconcern', 'insurgent', 'credit',
'displeased', 'get', 'violating', 'shepherd', 'refered', 'unprecedented',
'gender', 'certificate', 'publication', 'alleviated', 'unsighted', 'dysentery',
'needlepoint', 'abductor', 'deceitful', 'autopilot', 'enunciating', 'excessive',
'boastful', 'coffee', 'culture', 'preparation', 'sell', 'student', 'inexpert',
'subserve', 'desiccating', 'irredeemable', 'constitute', 'evaluate',
'unostentatious', 'performance', 'propagate', 'mushroomed', 'trouble',
'restraint', 'grocery', 'circumvent', 'match', 'book', 'unannounced',
'dreadnought', 'appearance', 'laugh', 'disagree', 'measure', 'subletting',
'body', 'unstuff', 'peace', 'excitations', 'concentration', 'international',
'Japanese', 'advised', 'astronomer', 'sound', 'deforming', 'material',
'fetishism', 'syntaxes', 'inorganic', 'counterfeit', 'devilishly', 'carry',
'mental', 'spot', 'restrainer', 'live', 'socialise', 'reassess', 'scholarship',
'foreclosed', 'prevent', 'missile', 'abuse', 'edible', 'germinate', 'imitate',
'criticality', 'initialise', 'cabbage', 'suspiciousness', 'fruitful', 'respite',
'sign', 'sweet', 'preconceptions', 'badness', 'tax', 'memorialize', 'tableware',
'tube', 'traitorous', 'hire', 'mouth', 'admitting', 'append', 'differentia',
'monotype', 'remind', 'immobile', 'interrelated', 'incommensurable', 'heavy',
'socialites', 'intermingles', 'Wednesday', 'voraciously', 'muster',
'insufficiency', 'england', 'minority', 'language', 'attack', 'undeviating',
'covering', 'replicate', 'unimpressed', 'impressionable', 'defiles', 'reform',
'oxide', 'skillfulness', 'bestowal', 'wheaten', 'real', 'castled', 'falsifier',
'attributable', 'mercantile', 'implausible', 'journey', 'site', 'gem',
'disturbances', 'enfolded', 'astonish', 'algebraic', 'available', 'rounded',
'repulses', 'vulnerable', 'unmelted', 'walloper', 'pedaler', 'cut',
'wilderness', 'sport', 'squishing', 'nakedness', 'district', 'inheritor',
'intragroup', 'transmigrating', 'argue', 'involvement', 'autoregulation',
'health', 'apparition', 'eventful', 'fauna', 'choose', 'area', 'trial',
'ethnic', 'ebb', 'similar', 'localise', 'analyzed', 'translocate',
'conspicuousness', 'triclinic', 'multidimensional', 'guillotine', 'network',
'exaction', 'entrench', 'scandinavian', 'building', 'obtainment', 'diagonal',
'sympathized', 'asian', 'bendability', 'substance', 'sorcery', 'kingship',
'absconding', 'kind', 'galvanize', 'artifact', 'spill', 'avenue', 'process',
'life', 'diffidence', 'hill', 'estrogenic', 'reproduction', 'vector', 'compel',
'changeableness', 'history', 'casteless', 'plundered', 'seafaring', 'brightly',
'page', 'situation', 'defiant', 'subspecies', 'chauvinist', 'room', 'chromatic',
'stupid', 'bank', 'lay', 'unwrap', 'kid', 'poisoning', 'animality',
'sublieutenant', 'transvestitism', 'rebellious', 'overshoe', 'crudeness',
'penis', 'tunneled', 'assessments', 'independences', 'short-change',
'conformations', 'distinction', 'censoring', 'dirty', 'fluidity', 'automatic',
'critic', 'untilled', 'discrimination', 'brisker', 'nonperformance',
'friendliness', 'unfavorable', 'habitable', 'bleach', 'paparazzo', 'scrutiny',
'crane', 'destroyers', 'pronounce', 'transformation', 'self-improvement',
'humanness', 'unisons', 'concordance', 'bengali', 'heat', 'resource',
'shepherded', 'thrombosis', 'trioxide', 'boldness', 'precedent', 'backwardness',
'cofactor', 'scandalize', 'enunciates', 'liveable', 'comfortable', 'deep',
'commissions', 'acceptance', 'embezzle', 'upset', 'replacements', 'evaporate',
'inessential', 'protestant', 'intercede', 'recommendation', 'organismal',
'animalism', 'principality', 'raw', 'combusting', 'transalpine', 'positioners',
'hike', 'reprints', 'nonconscious', 'obstructive', 'disadvantaged',
'infectiously', 'regenerate', 'note', 'interlink', 'naturalise', 'stewardship',
'gracefulness', 'decision', 'synchronic', 'consumptive', 'nontoxic', 'expose',
'greenness', 'procreated', 'despoil', 'thinness', 'entrust', 'repeat', 'nurse',
'terrorize', 'join', 'internationalize', 'dissociations', 'consubstantial',
'microcircuit', 'empower', 'disability', 'sexy', 'stockers', 'prize',
'prejudging', 'seepage', 'photographer', 'intertwining', 'delimited',
'transmigrated', 'combatted', 'religious', 'poison', 'shape', 'behaviorist',
'suspect', 'microfilm', 'suppressor', 'deflate', 'exchangeable', 'stitch',
'seller', 'five', 'afghani', 'scampering', 'flee', 'mammal', 'line', 'emerald',
'uncomfortable', 'enslaves', 'recitalist', 'governance', 'moderate',
'attendances', 'innovativeness', 'unparented', 'American', 'commandership',
'purveying', 'scientist', 'weaken', 'steepen', 'informal', 'technology',
'spread', 'deletion', 'universe', 'submarine', 'disquieting', 'disgust',
'breathe', 'uncomprehending', 'germanic', 'retraction', 'noon', 'translunar',
'orchestrations', 'unpersuasive', 'genuinely', 'detectable', 'lengthy',
'energetic', 'invigorating', 'negotiable', 'plate', 'survivalist', 'entombment',
'fundamentalism', 'bounce', 'microfossils', 'Harvard', 'develop', 'circumpolar',
'example', 'concurrencies', 'modesty', 'statement', 'consigning', 'cheap',
'hive', 'gene', 'weirdly', 'blitzed', 'performances', 'thousand', 'squirt',
'singe', 'desirable', 'unnecessary', 'effort', 'race', 'serve', 'venomous',
'highjacking', 'wrestle', 'serial', 'picture', 'unfeathered', 'lastingly',
'rack', 'rook', 'help', 'trace', 'metabolism', 'sociable', 'luxuriance',
'spacewalker', 'wheel', 'defeating', 'premisses', 'obviousness', 'condensing',
'association', 'fortitude', 'incommutable', 'system', 'lie', 'susceptible',
'force', 'standardize', 'brand-newness', 'enthuse', 'algebra', 'arrange',
'laboratory', 'branch', 'intense', 'bewitchment', 'check', 'computation',
'anamorphosis', 'subdividing', 'hybridise', 'shrieks', 'traditionalism',
'continue', 'repurchases', 'disclosure', 'program', 'unrepeatable', 'word',
'stroked', 'survey', 'beginning', 'cry', 'rotational', 'engrave', 'exhibited',
'confess', 'residing', 'warship', 'rudderless', 'insurrectional',
'unaccessible', 'content', 'cockerel', 'attested', 'woodland', 'calendar',
'secularist', 'painkillers', 'ganging', 'equivalence', 'interlayers',
'undetectable', 'assay', 'biographer', 'prophetic', 'standard', 'brandy',
'inabilities', 'lover', 'protectorship', 'employed', 'nonviable', 'distressful',
'prisoner', 'unformed', 'unicycling', 'contraception', 'transducers',
'codefendants', 'perform', 'sugar', 'thermonuclear', 'ill', 'microvolts',
'dissolved', 'murderer', 'lilt', 'colorful', 'official', 'painter', 'position',
'summonings', 'disjunct', 'outfoxed', 'flattery', 'fill', 'reenactor',
'breather', 'insensitive', 'hypersensitive', 'registry', 'scenery', 'poll',
'gravity', 'fruiterer', 'absence', 'astronomical', 'sermonize', 'problem',
'bootless', 'complimentary', 'placidity', 'scowl', 'fascinate',
'protectiveness', 'monk', 'preteens', 'disfavor', 'stroke', 'microcircuits',
'nomad', 'basketball', 'offensive', 'carburettors', 'largeness', 'preempt',
'fighting', 'supermarket', 'muscularity', 'perceptible', 'macroevolution',
'arab', 'flighted', 'reproachful', 'inducement', 'respectively', 'animalize',
'necessitate', 'keyboard', 'evidence', 'debarred', 'noble', 'sweetish',
'encrusted', 'submariners', 'negociate', 'initiate', 'rabbi', 'restrict',
'direction', 'blow', 'constancy', 'equality', 'defame', 'intending', 'swing',
'invitation', 'editor', 'madhouse', 'think', 'profit', 'possession',
'drownings', 'unfortunate', 'abandonment', 'virologist', 'aim', 'reasoning',
'radiance', 'excavations', 'remarkable', 'mechanical', 'accomplished',
'follower', 'prehistorical', 'important', 'insecurities', 'sponsor',
'difference', 'commode', 'pottery', 'skid', 'undefinable', 'monogram',
'circumvents', 'shortish', 'omission', 'icelandic', 'unintelligible',
'undiscerning', 'desire', 'fertility', 'change', 'satisfactory', 'reproves',
'authority', 'size', 'dissimulate', 'vaporise', 'surroundings', 'inflammation',
'eavesdropper', 'fever', 'postglacial', 'intercommunicate', 'self', 'possessor',
'infeasible', 'unclog', 'car', 'creative', 'retrace', 'managership',
'extension', 'extort', 'connect', 'lenience', 'embroideries', 'circumcising',
'secret', 'partner', 'convert', 'spoonfuls', 'cucumber', 'supposed',
'baptistic', 'intercession', 'categorization', 'immobilization', 'synchronized',
'structure', 'shackle', 'eye', 'science', 'impossible', 'delight', 'gladness',
'unsuitable', 'index', 'beat', 'sight', 'idiocy', 'underprivileged',
'companionships', 'unconscious', 'illiberal', 'attachment', 'radical',
'exclaiming', 'intelligences', 'unflagging', 'selectively', 'huffy',
'bastardize', 'misleading', 'benefited', 'noncitizens', 'reduce', 'prudery',
'extravert', 'toppled', 'postmodernist', 'adventism', 'eruptive', 'emulsify',
'incorrupt', 'subeditor', 'glass', 'liquid', 'discovery', 'player', 'drawers',
'unheralded', 'rustic', 'group', 'corrupt', 'feminised', 'maildrop',
'migrational', 'impermissible', 'predetermine', 'repel', 'relocation',
'enthusiastic', 'discountenance', 'company', 'engorge', 'religiousness',
'contravened', 'deadness', 'helm', 'transshipped', 'galvanic', 'acquisition',
'roosted', 'radiators', 'extraterrestrial', 'wild', 'hostility', 'asylum',
'perfectible', 'isosceles', 'composed', 'about', 'advancement', 'prophetical',
'transubstantiate', 'sentenced', 'unobjectionable', 'spiritualize',
'circumnavigations', 'encroachments', 'displeases', 'curvature', 'sexual',
'directional', 'approved', 'prisoners', 'law', 'designs', 'autobuses',
'blithering', 'career', 'therapeutical', 'label', 'trespass', 'crusaders',
'unmanned', 'unassertiveness', 'extinguish', 'integrity', 'internationaler',
'movie', 'contrarily', 'accordance', 'injure', 'antedating', 'spherical',
'fireproof', 'troops', 'acting', 'accessible', 'blessing', 'battleships',
'opinion', 'crier', 'flight', 'transfusing', 'professor', 'touch', 'expel',
'postmark', 'happiness', 'reasonable', 'organic', 'encapsulate', 'improving',
'unisexual', 'secretary', 'Arafat', 'amorphous', 'equip', 'interact', 'travel',
'cell', 'victim', 'conclusive', 'drink', 'speed', 'noisy', 'expressionless',
'series', 'subordination', 'animal', 'encouragement', 'bridge', 'entrance',
'exacted', 'knowing', 'fractionate', 'internationality', 'accommodation',
'image', 'nonfunctional', 'inmate', 'excitation', 'acrobat', 'hundred', 'motto',
'elated', 'dictatorship', 'evolution', 'withdrawal', 'functionality',
'corpulence', 'distributive', 'supernatural', 'depopulate', 'ship',
'monoculture', 'unquenchable', 'pathless', 'dangerous', 'rectorate',
'duplicable', 'literalness', 'run', 'bird', 'freakishly', 'surround',
'hospitalize', 'federalize', 'cheapen', 'tiger', 'conflagration', 'stimuli',
'comprehensive', 'inoffensive', 'ceaseless', 'militarize', 'designed',
'refinery', 'washers', 'disestablishing', 'preconception', 'resurfacing',
'spangle', 'homogenized', 'volunteer', 'agitation', 'board', 'careerism',
'farmer', 'undisputable', 'numerical', 'earmuffs', 'inscribe', 'virtuoso',
'respectable', 'besieging', 'imperils', 'know-how', 'partnership', 'clownish',
'differences', 'anticancer', 'assassinated', 'heterosexism', 'exterminator',
'reordering', 'unilateralist', 'utterance', 'goldplated', 'dematerialised',
'gelatinous', 'motivation', 'surpass', 'interesting', 'classify', 'fire', 'pay',
'homogeneous', 'cynically', 'scot', 'fabricate', 'topically', 'scope', 'film',
'unicyclist', 'monogenesis', 'device', 'postdates', 'infrastructure',
'authorship', 'immortalize', 'unsatisfactory', 'general', 'leverage', 'current',
'languishing', 'remitting', 'fight', 'insure', 'cocoon', 'ripeness', 'colonise',
'different', 'separationist', 'hop', 'election', 'connection', 'puritanism',
'academicism', 'demanded', 'guidance', 'bishop', 'carbonic', 'center',
'capture', 'traveler', 'outshout', 'unquestioned', 'creator', 'proton',
'quitter', 'jaguar', 'convector', 'disarranged', 'embroideress', 'disaster',
'septic', 'unexpected', 'property', 'infolding', 'galaxy', 'distinguishing',
'objectify', 'lusterware', 'enrollment', 'starkness', 'internet', 'splitter',
'supplanting', 'abnormality', 'deposit', 'valor', 'indoctrinate', 'grassroots',
'traversals', 'ordain', 'kilometer', 'rhymers', 'refurbishments', 'military',
'concert', 'merchandise', 'circumventing', 'execute', 'cofactors', 'transact',
'effectiveness', 'quadratics', 'agency', 'tournament', 'quicken', 'stoical',
'approachable', 'unrewarding', 'day', 'territory', 'confine', 'guest',
'sprinkle', 'inexpedient', 'regimental', 'undefeated', 'replications',
'obvious', 'elaborate', 'clozapine', 'museum', 'pave', 'depression', 'server',
'unintelligent', 'noise', 'black', 'slanderous', 'party', 'condition', 'focus',
'freshen', 'planet', 'hypertension', 'reduced', 'precociously', 'angrier',
'pitch', 'resides', 'cooperation', 'hospital', 'ostentatious', 'clericalism',
'christianise', 'attacker', 'deviationism', 'indicted', 'hilarity', 'invisible',
'fuck', 'lushness', 'commutation', 'deformity', 'tennis', 'unsalable',
'sportive', 'resistive', 'blunders', 'helical', 'dominance', 'urbanize',
'ecology', 'preposed', 'southern', 'rumbled', 'imposition', 'warning', 'mogul',
'divide', 'kindergarteners', 'lend', 'world', 'heater', 'edgeless', 'Jerusalem',
'aerialist', 'descend', 'internships', 'caramelize', 'protrusion', 'reckoner',
'inclosure', 'laud', 'rock', 'inconvertible', 'favourable', 'refuted',
'mistrustful', 'unmolested', 'transponder', 'critical', 'epicure', 'practice',
'harden', 'jarringly', 'case', 'Freud', 'seasonable', 'primer', 'long',
'predators', 'specialism', 'seriousness', 'uninformed', 'cynical', 'omnipotent',
'eat', 'improvement', 'fringes', 'nightly', 'inharmonious', 'inroad', 'popcorn',
'magically', 'convocation', 'domain', 'incalculable', 'hypercoaster',
'socialist', 'monograms', 'heedless', 'imitation', 'embody', 'brood',
'unilluminated', 'strengthened', 'muscle', 'stand-in', 'database',
'institutionalize', 'manner', 'star', 'disassembled', 'skidding', 'entity',
'marketers', 'papered', 'depreciate', 'reinterpret', 'unforgiving',
'horsemanship', 'Mars', 'friendship', 'interlingua', 'put', 'importance',
'conductance', 'attainment', 'labourer', 'evangelize', 'cardinality',
'consciousness', 'adapt', 'causing', 'knifing', 'impeded', 'indexical',
'unloved', 'classicist', 'kill', 'latinist', 'collection', 'expounded',
'microphallus', 'balance', 'marginalize', 'untroubled', 'interspecies',
'employments', 'classification', 'characterless', 'combination', 'salable',
'disfavoring', 'belief', 'encouraging', 'Yale', 'disassociates', 'unknowing',
'canvass', 'battened', 'acknowledgement', 'illiterate', 'personifying',
'recorders', 'cowboys', 'wine', 'baggers', 'naivety', 'gardens', 'dooming',
'forest', 'direct', 'discounters', 'sing', 'unskillfulness', 'cross-index',
'handbook', 'perfect', 'reclassifications', 'hypersensitivity', 'electrical',
'subserving', 'ruralist', 'regained', 'viewer', 'extraterrestrials',
'irremovable', 'transmitter', 'fuel', 'bounced', 'foreign', 'letter', 'buck',
'impotently', 'explorers', 'antitoxic', 'dig', 'growth', 'ruler', 'street',
'viscometry', 'contravene', 'victory', 'unblock', 'tail', 'romanic', 'wizard',
'intraspecific', 'moon', 'combusts', 'disbelieving', 'cofounders', 'earning',
'illegal', 'immobilizing', 'order', 'vindictiveness', 'become', 'quality',
'unicycles', 'halfhearted', 'demerit', 'exclamation', 'characteristic',
'politician', 'circumference', 'selling', 'fulfillments', 'nonpolitical',
'game', 'investor', 'sheepish', 'gloom', 'autocracy', 'partnerships',
'concreteness', 'antipsychotic', 'unzipping', 'usher', 'heartlessness',
'disorderly', 'rareness', 'cosponsoring', 'encoded', 'directionless',
'instruct', 'uncontrolled', 'nonconformist', 'discipline', 'yellowish',
'magnetic', 'subsurface', 'unbiased', 'requests', 'broadcasters', 'block',
'ticket', 'membership', 'weapon', 'sexism', 'tailgate', 'king', 'intersected',
'empty', 'hotel', 'uncommunicative', 'adaptive', 'actor', 'software',
'circumscribed', 'remember', 'transsexual', 'leadership', 'profitless',
'interest', 'interdisciplinary', 'primates', 'uninterested', 'abnormal',
'arouse', 'apolitical', 'spouse', 'sandwich', 'ennobled', 'concerti', 'enjoins',
'rooters', 'dissociable', 'acoustics', 'unreserved', 'dry', 'fantasist',
'frighten', 'presenting', 'discourteous', 'representational', 'fasten',
'parallelize', 'prejudge', 'cession', 'pledged', 'foresters', 'narrow-minded',
'distillate', 'preaching', 'censorships', 'seat', 'food', 'disengages',
'interlace', 'headless', 'sessions', 'subtropical', 'racket', 'reviewers',
'insidiously', 'spiritualist', 'banished', 'funeral', 'presence', 'profanity',
'strife', 'insurance', 'algebras', 'issue', 'football', 'unaffected', 'situate',
'imprecise', 'unprofessional', 'refresher', 'concurrency', 'incontestable',
'nerveless', 'worsens', 'regulate', 'legion', 'row', 'splice', 'behavioural',
'supplement', 'itch', 'transportation', 'letters', 'seeders', 'incubate',
'rhythmicity', 'theater', 'dissonance', 'prayer', 'racism', 'americanize',
'bellowing', 'populace', 'procurators', 'document', 'unwed', 'coeducation',
'cooperators', 'enhancement', 'skiing', 'moderatorship', 'inaccessible',
'disloyal', 'guardedly', 'adverse', 'finality', 'inheritable', 'increase',
'conjoins', 'secluding', 'hold', 'envelop', 'autograft', 'causative', 'smooth',
'equipment', 'uncertainty', 'provisionary', 'accommodative', 'microwaving',
'developments', 'disrespectful', 'reply', 'illimitable', 'circumvented', 'mind',
'commodes', 'astronautical', 'dependence', 'disk', 'confirmable', 'affirm',
'antechamber', 'guarantee', 'skilled', 'digitise', 'adjournment', 'contrive',
'marker', 'potent', 'postcode', 'vegetational', 'containers', 'undated',
'rehashing', 'perfective', 'approach', 'mingles', 'artlessness', 'government',
'currency', 'vindictively', 'royalist', 'unfavourable', 'postmodernism',
'memoir', 'founder', 'similarity', 'demoralise', 'flatulence', 'utilitarianism',
'binging', 'blurting', 'remounted', 'friendships', 'major', 'forecast', 'steep',
'analogous', 'unworthiness', 'duty', 'variation', 'interwove', 'nanosecond',
'confinement', 'urgency', 'radio', 'travelers', 'defrayed', 'outlawed',
'discriminatory', 'infectious', 'care', 'grinder', 'alarmism', 'extrajudicial',
'reproducible', 'analyze', 'talkativeness', 'command', 'extending', 'wealthy',
'circumspect', 'penitent', 'sprint', 'play', 'impolitic', 'fear', 'declare',
'synthesize', 'confluent', 'clergyman', 'italian', 'unacceptable', 'unsettled',
'percent', 'patrol', 'scattered', 'lubricate', 'robbery', 'educate', 'dark',
'hallucinating', 'guard', 'disavowed', 'unspecialised', 'interlaces', 'rub',
'hazard', 'longing', 'write', 'sit', 'ukrainians', 'censorship', 'intramural',
'love', 'Mexico', 'autobiographer', 'forbid', 'reinsured', 'music',
'revolutionise', 'humorous', 'incredulous', 'monarchical', 'gin', 'uproarious',
'reformism', 'ungraceful', 'pressurise', 'discordance', 'talk', 'freighter',
'victorious', 'corrode', 'tricolor', 'crisis', 'macroeconomist', 'publicise',
'dissenter', 'appraisal', 'large', 'coat', 'entertainer', 'merchantable',
'small', 'highlanders', 'bite', 'mathematician', 'retarding', 'posthole',
'playful', 'secondary', 'plant', 'abundance', 'enchantress', 'sufficed',
'untracked', 'predictive', 'undesirable', 'baste', 'queen', 'punctuate',
'children', 'expounding', 'observe', 'wealth', 'freshness', 'oil',
'championship', 'announcement', 'crispness', 'protraction', 'cliffhanger',
'interceptor', 'possess', 'postponements', 'eroticism', 'start', 'ejector',
'commit', 'listing', 'slack', 'snooper', 'autosuggestion', 'weaponize',
'figurative', 'magician', 'inquiring', 'impoliteness', 'emigration',
'acquisitive', 'mildness', 'thatcher', 'innocuous', 'anger', 'roofers', 'lease',
'consign', 'reputable', 'standing', 'hush', 'lithium', 'nonindulgent',
'harmful', 'semiconducting', 'practical', 'burying', 'environment', 'puffery',
'unloving', 'mutinied', 'prominence', 'microbiologist', 'criticism',
'enforcing', 'banquet', 'ear', 'interlinking', 'inheritances', 'paragraph',
'hydrochloride', 'characters', 'giving', 'authorize', 'spellers', 'syntactic',
'inquisitive', 'title', 'follow', 'greengrocery', 'conformism', 'insatiate',
'undemocratic', 'interpreter', 'immigrate', 'skateboarders', 'doctrine',
'unwanted', 'ascendence', 'zoo', 'advisory', 'dissenters', 'irritatingly',
'malevolence', 'believing', 'pleading', 'perceive', 'inbreeding',
'extraterritorial', 'irrationality', 'unfledged', 'unmarketable', 'atmosphere',
'shoot', 'carbonate', 'recycle', 'embellishment', 'wrathful', 'antifeminist',
'disguise', 'aid', 'psychodynamics', 'mother', 'brotherhood', 'philosophic',
'physically', 'collected', 'antitumor', 'postdated', 'broad', 'witch-hunt',
'move', 'unploughed', 'autobiographies', 'campfires', 'singing', 'ceramicist',
'self-discovery', 'telephone', 'object', 'emotional', 'inexplicable',
'brainless', 'vicarious', 'opportune', 'CD', 'symmetrical', 'organism',
'automobile', 'profusion', 'hover', 'link', 'rearrangements', 'kidnapped',
'skater', 'softness', 'depictive', 'down', 'regardless', 'term', 'exceedance',
'deviously', 'postposition', 'excitements', 'recast', 'denominate', 'sniffers',
'cosigns', 'amateurish', 'tea', 'dispossess', 'flightless', 'pestilence',
'assistance', 'preschooler', 'transposable', 'subfamily', 'diagonals',
'stickler', 'defensive', 'unwelcome', 'intermarry', 'shrewdness', 'regretful',
'monarchic', 'morality', 'fence', 'cofounder', 'resistor', 'excitedly',
'inquisitiveness', 'thing', 'spend', 'report', 'stay', 'remainder', 'lightship',
'perished', 'postholes', 'hypervelocity', 'enlist', 'investigator',
'condescend', 'asphaltic', 'incommensurate', 'imperfection', 'diver', 'inform',
'aspirate', 'interpreted', 'people', 'phenomenon', 'irrelevant', 'continence',
'gringo', 'cheat', 'computer', 'musical', 'graft', 'dam', 'extraversion',
'hypermarkets', 'midday', 'explain', 'drought', 'source', 'indiscriminate',
'unchaste', 'decorate', 'heraldist', 'exacerbated', 'prescriptions', 'physics',
'untrustworthy', 'plucked', 'tricolour', 'competition', 'promiscuous',
'devilish', 'impassively', 'speculate', 'preservation', 'reliable', 'entraps',
'lesson', 'amazings', 'assigned', 'connoting', 'sink', 'future', 'voice',
'hydrolysed', 'industry', 'titillated', 'tenured', 'pick', 'automate',
'medicate', 'indifferently', 'nonpublic', 'mccarthyism', 'ineffective',
'deserters', 'explorer', 'reviles', 'impulsion', 'potential', 'encamping',
'unarguable', 'interweaved', 'ringer', 'run-down', 'heiress', 'dishonest',
'hormone', 'clamorous', 'calculate', 'investigation', 'exempt', 'complain',
'demand', 'nonprofessional', 'devalue', 'support', 'price', 'reciprocal',
'holder', 'decay', 'monoclinic', 'sourdough', 'autoimmune', 'limit', 'pretense',
'hateful', 'separate', 'right', 'princedoms', 'significances', 'purposeless',
'give', 'intelligent', 'incongruous', 'proximity', 'wrongdoer', 'brandish',
'unfasten', 'auditive', 'preservers', 'medicine', 'suppleness', 'quieten',
'read', 'embroiderer', 'fieldworker', 'annoy', 'actuator', 'landscape',
'excrete', 'arbitrary', 'formations', 'suppress', 'aqueous', 'contest',
'inaccurate', 'indispensable', 'marathon', 'historically', 'subhead', 'trading',
'virility', 'insanity', 'inconsiderate', 'space', 'reformations',
'fragmentation', 'thick', 'contrastive', 'unsubdivided', 'inflection', 'code',
'interjection', 'obstruct', 'experimenter', 'observation', 'psychiatry',
'dimensional', 'boy', 'aluminum', 'separatist', 'psychologist', 'coupling',
'meaningless', 'londoners', 'grassland', 'rebel', 'nonrepresentational',
'century', 'Brazil', 'necessary', 'calmness', 'strangers', 'enlarger',
'predominance', 'hunt', 'meadows', 'intelligence', 'attempt', 'soloist',
'traverse', 'statistician', 'unequivocal', 'capitation', 'anticyclones', 'FBI',
'rubberstamp', 'confidence', 'mathematical', 'burn', 'frowning', 'adulteration',
'security', 'compartmentalization', 'rascality', 'vodka', 'together',
'aeronautical', 'antagonist', 'baseness', 'whizzed', 'construct',
'consequences', 'cosponsors', 'psychology', 'result', 'Israel', 'wisdom',
'piety', 'monoatomic', 'active', 'expressible', 'employable', 'industrialise',
'buy', 'bodily', 'cognizance', 'disprove', 'impartiality', 'unsexy',
'sternness', 'ravenous', 'sustainable', 'news', 'monocultures', 'carrier',
'machine', 'creation', 'lively', 'genre', 'lectureship', 'save', 'allergic',
'discipleship', 'internee', 'preserve', 'opalescence', 'hyperlink',
'decelerate', 'connectedness', 'disgruntle', 'cover', 'intracerebral',
'repositioned', 'repeating', 'reserve', 'subgroup', 'shanghai', 'autografts',
'protesters', 'nurturance', 'gravitated', 'tidings', 'interlinks', 'transfused',
'anaesthetics', 'misbehave', 'philanthropy', 'undissolved', 'crouch',
'hypothesis', 'venders', 'warrior', 'significant', 'wicked', 'sea',
'abbreviate', 'cemetery', 'past', 'implication', 'comport', 'settle',
'anarchist', 'proposition', 'autographic', 'chip', 'unionise', 'buggered',
'chooses', 'mimicked', 'coinsurance', 'convertible', 'conscripting',
'scheduled', 'corral', 'maker', 'sufferance', 'unicycle', 'presenters', 'allow',
'invite', 'arrangement', 'transfigure', 'confinements', 'frequency', 'voyage',
'control', 'angular', 'reprehensible', 'heartlessly', 'education', 'decoration',
'macroeconomists', 'refuels', 'reporters', 'point', 'practicality', 'empirical',
'breed', 'crystalline', 'unceremonious', 'hotness', 'sponge', 'please',
'trader', 'commingle', 'autoerotic', 'scarcity', 'mitigated', 'slacken',
'priest', 'encyclopaedic', 'assign', 'unmarried', 'team', 'deceiver',
'serenaded', 'interview', 'resigning', 'engineering', 'intramuscular', 'widen',
'circularize', 'dawn', 'stressor', 'homophobia', 'Jackson', 'short', 'validate',
'insecureness', 'consumer', 'energy', 'explicit', 'irrigate', 'astringe',
'smoothen', 'inanimate', 'delimitations', 'unconcerned', 'strength',
'pronunciation', 'wingless', 'credibility', 'loveless', 'coiled', 'immoveable',
'impurity', 'sensitivity', 'representable', 'discriminating', 'foreigners',
'intended', 'disabused', 'synoptic', 'lavishness', 'rite', 'baby', 'sailings',
'fleshiness', 'deconstruct', 'list', 'push', 'producing', 'assimilate',
'sidewinder', 'hardware', 'continuously', 'entreaty', 'probability',
'preordained', 'feline', 'wrongdoing', 'desertion', 'fixture', 'communicator',
'prideful', 'pious', 'normalise', 'cross-link', 'opposition', 'quarter',
'antonymous', 'skin', 'assessment', 'jewel', 'recovery', 'migrate',
'corespondent', 'office', 'discolor', 'cup', 'learn', 'administration', 'cost',
're-create', 'circle', 'status', 'doubt', 'credentials', 'communistic',
'rattlesnake', 'premise', 'request', 'enjoining', 'address', 'autofocus',
'refurbishment', 'clarify', 'palestinians', 'layer', 'state', 'ordinary',
'cuteness', 'roosters', 'foreigner', 'nanometer', 'negligence', 'airship',
'subspaces', 'drug', 'train', 'subdivided', 'librarianship', 'canker',
'knightly', 'momentousness', 'reprocessing', 'preliterate', 'omniscience',
'sincere', 'jazz', 'transfer', 'inch', 'helplessness', 'cognition', 'cosigned',
'shift', 'intensions', 'comfort', 'receiverships', 'schemer', 'indirect',
'contrabands', 'defeatist', 'exciting', 'molar', 'reproductive', 'treat',
'draw', 'comment', 'delay', 'ministry', 'generation', 'personnel', 'undefended',
'interrelationship', 'circumferential', 'tasty', 'intend', 'snookered',
'apologize', 'normalize', 'randomize', 'bunking', 'manslaughter', 'replication',
'consonant', 'fruit', 'bureaucrat', 'worker', 'high', 'fossil', 'territorials',
'microflora', 'intoxication', 'donate', 'blackmailed', 'polar', 'fingerprint',
'communication', 'departure', 'populate', 'waste', 'prospector', 'tripods',
'swan', 'inadvertence', 'string', 'iranian', 'ulcerate', 'cozy', 'architecture',
'early', 'balanced', 'record', 'out', 'improvise', 'lad', 'hankering', 'quote',
'enshrouded', 'call', 'purgatory', 'coerce', 'provincialism', 'pathfinder',
'contraries', 'disgorge', 'year', 'enforcements', 'established', 'associations',
'retrials', 'adhesion', 'involve', 'microbalance', 'political', 'irreligious',
'ceremonious', 'coefficient', 'asexual', 'tasteless', 'primacy', 'nonobservant',
'nerve', 'gibberish', 'interconnectedness', 'pathfinders', 'concerts', 'brown',
'airplane', 'register', 'maleness', 'self-discipline', 'shock', 'inference',
'dynastic', 'pilot', 'narrow-mindedness', 'untruth', 'install', 'monoplanes',
'wholeheartedness', 'faze', 'snickering', 'endangerment', 'closet',
'impermanent', 'benefactor', 'video', 'liability', 'chord', 'unapproachable',
'christianity', 'choke', 'mount', 'imply', 'acquiring', 'local',
'businessperson', 'conclusion', 'symbol', 'evil', 'oracle', 'diarrhea', 'bed',
'orientate', 'toxic', 'circumstances', 'gauge', 'reason', 'sex', 'reenact',
'absorbing', 'homoerotic', 'balminess', 'gathered', 'paper', 'deflowering',
'independently', 'irreverence', 'remakes', 'convergent', 'picket',
'electioneering', 'leisured', 'oppress', 'forceps', 'opera', 'constant',
'tolerable', 're-argue', 'department', 'persuasions', 'conscientious',
'communicativeness', 'morph', 'trilateral', 'loose', 'information',
'premeditation', 'spaciousness', 'emotionalism', 'rich', 'subdivide', 'bush',
'assistances', 'design', 'being', 'tend', 'library', 'suffer', 'lawyer',
'extractor', 'family', 'brittany', 'preheated', 'economic', 'hit', 'inelegance',
'attitude', 'kazakhstani', 'admit', 'dazzle', 'over', 'tricolours', 'summer',
'memorabilia', 'enroll', 'quarrel', 'passable', 'discontinuance', 'enliven',
'endorse', 'roller', 'deal', 'disembodied', 'tricycle', 'gain',
'incontrovertible', 'irresolution', 'ecclesiastic', 'operation', 'transvestite',
'canonize', 'brained', 'relates', 'tie', 'phosphate', 'successful', 'giant',
'paving', 'chairmanship', 'antifeminism', 'impossibilities', 'schnauzer',
'perspectives', 'exemplify', 'withhold', 'footballers', 'implement', 'tool',
'performing', 'incomprehension', 'sexless', 'antipsychotics', 'docile', 'knock',
'automates', 'population', 'combusted', 'dispersive', 'fiddled',
'intramolecular', 'commingled', 'spiciness', 'hard-and-fast', 'war',
'subroutines', 'preassembled', 'excommunicate', 'fly', 'locality', 'chemistry',
'bicycle', 'produce', 'trade', 'converse', 'hinduism', 'slaughterers',
'ashamed', 'discernment', 'soulfully', 'distinguish', 'removal', 'combustion',
'needleworker', 'portrayer', 'exclusive', 'relieve', 'territorial', 'bobbers',
'express', 'toss', 'whimsically', 'woman', 'plane', 'media', 'storm',
'researchers', 'physical', 'holy', 'adversely', 'teaspoonful', 'amusements',
'uninhibited', 'associate', 'greeting', 'juncture', 'embroiderers', 'vacations',
'agreement', 'agent', 'covered', 'individualist', 'concavity', 'wordless',
'immeasurable', 'hash', 'shout', 'noncivilized', 'determine', 'archive',
'legal', 'organization', 'garment', 'concerning', 'subsequences', 'subtend',
'spirited', 'workman', 'Palestinian', 'suggestible', 'condescended',
'conjurors', 'apply', 'accomplishments', 'flag', 'evangelistic', 'copulate',
'exist', 'separation', 'charge', 'boxing', 'formalisms', 'phone',
'objectifying', 'undatable', 'heading', 'monopolist', 'blend', 'photocopy',
'development', 'indelicate', 'immensely', 'filing', 'cylindric', 'sitting',
'writer', 'treatment', 'antagonize', 'inducted', 'loveable', 'shanked', 'plot',
'bench', 'magnetize', 'microfiche', 'decide', 'practicable', 'supply',
'intrude', 'place', 'compound', 'penetrate', 'acoustic', 'spiritless', 'stump',
'synthetical', 'break', 'bluejacket', 'clothes', 'periodical', 'winners',
'fording', 'advocate', 'range', 'unshaped', 'evangelicalism', 'transfuse',
'abstractionist', 'notebook', 'incurved', 'deity', 'same', 'instructorship',
'micrometer', 'rooster', 'parallelism', 'soccer', 'calcify', 'speculativeness',
'conformity', 'princedom', 'facilitation', 'discrete', 'admission',
'uncontroversial', 'irregardless', 'rampant', 'flim-flam', 'project',
'reelections', 'tenderize', 'craftsman', 'look', 'tripod', 'malfeasance',
'gaiety', 'up', 'unmentionable', 'florescence', 'contagious', 'sanctioned',
'destroy', 'regionalisms', 'honor', 'rededicated', 'strengthen', 'prompt',
'prolapse', 'extendible', 'expensiveness', 'work', 'circumnavigate',
'nominated', 'standardise', 'minister', 'formal', 'exterminated', 'angry',
'clinic', 'performer', 'wear', 'feudalism', 'calculation', 'hypermarket',
'potato', 'incorruptible', 'temptation', 'expert', 'amounted', 'outperforming',
'receiving', 'moisten', 'archery', 'expansion', 'satisfaction', 'meaning',
'remove', 'placement', 'convene', 'clients', 'receptions', 'liberation', 'wash',
'man', 'sprouting', 'clamor', 'apocalyptical', 'weather', 'entrapped',
'fictitiously', 'inapplicability', 'nazi', 'brother', 'hypertexts',
'sophisticate', 'standoffish', 'carefreeness', 'reconstructs', 'willingness',
'fashionable', 'defeat', 'article', 'encrust', 'tumble', 'OPEC', 'piquancy',
'unproductive', 'unite', 'marinate', 'teasingly', 'copilots', 'powerful',
'imperceptible', 'entwined', 'reclaim', 'virginals', 'providence', 'parameter',
'warmness', 'thunderstorm', 'masculinity', 'overlying', 'vanish', 'undress',
'sightedness', 'deface', 'disinflation', 'rediscovery', 'copying', 'category',
'top', 'apprenticeship', 'frivolously', 'customise', 'proceeding', 'concerto',
'comfortless', 'informative', 'drive', 'reporter', 'action', 'seize',
'vocalism', 'winking', 'criterion', 'type', 'uproariously', 'flood', 'round',
'exporters', 'reassuringly', 'congruity', 'economist', 'even', 'incoordination',
'transmutes', 'fighter', 'undisclosed', 'oppose', 'interviewing', 'comparing',
'logical', 'cultist', 'reorientate', 'matter', 'volatility', 'interstellar',
'psychic', 'gathering', 'riskless', 'disposition', 'sanskrit', 'bibliographies',
'cargo', 'planning', 'increasing', 'isolation', 'headship', 'essential',
'verbalize', 'suspenseful', 'covert', 'private', 'wreathe', 'competes',
'generalized', 'coast', 'unattainableness', 'discontinuous', 'embracement',
'detail', 'dollar', 'inquisitor', 'new', 'compatibility', 'slavic', 'voter',
'healthful', 'yodeling', 'reassessments', 'replaces', 'discoverys', 'church',
'ignorance', 'money', 'emulsifying', 'promised', 'interconnect', 'stamp',
'antisubmarine', 'constitutive', 'appearances', 'effected', 'dwarfish',
'scratch', 'relation', 'convict', 'relationship', 'destabilization',
'internationalisms', 'stimulation', 'slave', 'correspondence', 'eldership',
'macrocosmic', 'wanderers', 'microseconds', 'confide', 'objector',
'reservation', 'microorganism', 'amethysts', 'rompers', 'pregnancy', 'heavenly',
'allurement', 'excretion', 'discharged', 'inheritance', 'catalogued', 'brain',
'deprive', 'shore', 'reproduce', 'repress', 'relinquishment', 'colored',
'decompositions', 'cat', 'sensualist', 'prayerful', 'inflicted',
'preadolescent', 'thoughtless', 'absolute', 'greenly', 'repositions',
'nonverbally', 'baseball', 'trio', 'extrasensory', 'splashy', 'surprise',
'beverage', 'sorrowful', 'pour', 'arrival', 'analogize', 'newness', 'fast',
'number', 'elector', 'insertion', 'containership', 'unenthusiastic',
'absorbance', 'cuddle', 'experience', 'caliper', 'insubordinate', 'crosswise',
'interned', 'gluttonous', 'individual', 'preheating', 'dissatisfying',
'demureness', 'stock', 'density', 'disapproving', 'mayoralty', 'invariable',
'sulfide', 'scar', 'stormy', 'immoderate', 'surface', 'remarriage', 'caesarism',
'compose', 'admittance', 'registration', 'energized', 'chronologize',
'deployment', 'cowered', 'juvenile', 'considerable', 'challenge', 'compare',
'interrelate', 'plan', 'sterile', 'methodically', 'liverpools', 'originality',
'lordship', 'hilariously', 'aspect', 'attendance', 'deduce', 'regularize',
'beauty', 'remedy', 'extraordinary', 'unreal', 'person', 'seed', 'ablaze',
'chloride', 'graveyard', 'acceptable', 'scornful', 'worsen', 'form', 'thrust',
'manifestation', 'dead', 'idle', 'exploitive', 'spatiality', 'odorize',
'unconsciousness', 'bring', 'egg', 'globalise', 'imbedding', 'hypothetical',
'enfolding', 'bright', 'unfit', 'abashed', 'implantations', 'adjustor',
'optical', 'improver', 'senate', 'unforeseen', 'circumcision', 'supporters',
'syllable', 'propriety', 'young', 'excommunicated', 'democratize', 'gift',
'affordable', 'motion', 'insurrectionist', 'malicious', 'protractors',
'journal', 'belligerence', 'irrelevance', 'sanctify', 'seductive', 'ideality',
'punjabi', 'management', 'impregnate', 'predetermination', 'mercifulness',
'dependent', 'stove', 'flavourful', 'mark', 'airport', 'nonnative', 'curved',
'believe', 'observed', 'chemical', 'worthy', 'deregulating', 'mechanism',
'presuppose', 'unrealizable', 'disjoined', 'extrapolations', 'trilogies',
'subjugate', 'incised', 'wood', 'subarctic', 'Maradona', 'distrust',
'anterooms', 'mar', 'divided', 'transmuted', 'yen', 'smallish', 'inelasticity',
'distress', 'planners', 'proud', 'symbolist', 'parasitical', 'indecent',
'autoloading', 'griping', 'enunciated', 'trichloride', 'scholarships',
'flicker', 'part', 'dutch', 'card', 'nondescripts', 'act', 'awkward',
'transgress', 'portioned', 'decomposition', 'killer', 'revivalism', 'tense',
'inquirer', 'unconvincing', 'unvariedness', 'attackers', 'promotive',
'constrict', 'unwaveringly', 'comprehensible', 'affect', 'nonpartisan',
'composure', 'box', 'associational', 'clairvoyant', 'stimulates', 'corruptive',
'uncreative', 'irreproducible', 'locate', 'nobelist', 'ropewalker', 'cook',
'monarchist', 'brace', 'glistens', 'cooperator', 'retraced', 'laureate',
'reburial', 'lieutenant', 'conductive', 'repressing', 'protestantism',
'emergency', 'rematches', 'deceive', 'lustrate', 'discover', 'television',
'interchanging', 'dividend', 'institution', 'attractor', 'depressor',
'individualize', 'circumvolution', 'color-blind', 'acoustical', 'requirement',
'homophony', 'constellation', 'religion', 'braid', 'incomprehensible',
'sulfuric', 'pastorship', 'canonical', 'slurred', 'annihilator', 'concurrence',
'severer', 'swooshing', 'enthusiast', 'syphons', 'nominate', 'bestowals',
'safe', 'spoonful', 'personify', 'seven', 'transmissible', 'valorous',
'scarceness', 'sponsorship', 'microcomputers', 'procreating', 'continuous',
'antecedent', 'hear', 'decrease', 'disease', 'extroversive', 'laundering',
'subjoined', 'translocation', 'wallpapered', 'impartial', 'flaunt',
'sanctifying', 'monsignori', 'nosiness', 'equatorial', 'rarity', 'hungry',
'dull', 'followed', 'rid', 'procedure', 'isolate', 'written', 'sociability',
'purchasable', 'applaud', 'removes', 'oust', 'returning', 'regularise',
'nature', 'copartnership', 'prejudice', 'consultive', 'president', 'divisible',
'water', 'smart', 'postmarks', 'independent', 'chatter', 'commenting',
'totalism', 'strains', 'governor', 'nonstandard', 'grievous', 'fancy',
'lobster', 'roadless', 'expound', 'incombustible', 'country', 'soulless',
'procurator', 'spiritize', 'plague', 'pedicab', 'arousal', 'univocal',
'inflame', 'conditional', 'autoimmunity', 'float', 'autopilots', 'blacken',
'torment', 'postcodes', 'gumption', 'addiction', 'bohemia', 'postboxes',
'activity', 'crew', 'unambiguous', 'superficial', 'disturbance', 'disfigure',
'inarticulate', 'girl', 'raise', 'born', 'copilot', 'steal', 'luxury',
'exceptionally', 'hyperlinks', 'indian', 'pittance'}

def manual_stem(word):
    if word[-2:] == 'ly':
        return word[:-2]
    elif word[-3:] == 'ish':
        return word[:-3]
    else:
        return None


added_words = []
vocab = {}
stemmed_vocab = {}
unk_vector = ""
ps = PorterStemmer()

for row in open(args.emb_path):

    word, *vec = row.split()
    vocab[word] = ' '.join(vec)
    stemmed_word = ps.stem(word)
    stemmed_vocab[stemmed_word] = ' '.join(vec)

    if word in words:
        print(row, end = "")
        added_words.append(word)
    elif word == '<UNK>':
        unk_vector = row

counta = 0
countb = 0
countc = 0
countd = 0
counte = 0
county = 0
countz = 0

for w in words:
    if w not in added_words:
        vec = None
        # print(f'{w} not in vocab')
        found = False

        # find stemmed word in vocab
        stemmed_word = ps.stem(w)
        if stemmed_word in vocab:
            vec = w + ' ' + vocab[stemmed_word] + '\n'
            # print(f'!!found stemmed {word}')
            found = True
            countd += 1
        
        if not found:
            if stemmed_word in stemmed_vocab:
                vec = w + ' ' + stemmed_vocab[stemmed_word] + '\n'
                # print(f'!!found stemmed {word}')
                found = True
                county += 1
        
        manual_stemmed_word = manual_stem(w)
        if not found:
            if manual_stemmed_word and manual_stemmed_word in vocab:
                vec = w + ' ' + vocab[manual_stemmed_word] + '\n'
                # print(f'!!found manually stemmed {manual_stemmed_word} for {w}')
                found = True
                countz += 1

        # find synonyms in vocab
        synonyms = wn.synsets(w)
        stemmed_synonyms = wn.synsets(stemmed_word)
        addl_stemmed_synonyms = []
        if manual_stemmed_word:
            addl_stemmed_synonyms = wn.synsets(manual_stemmed_word)
        synonyms += stemmed_synonyms + addl_stemmed_synonyms
        if not found:
            for synonym in synonyms:
                word = synonym.lemmas()[0].name()
                if word in vocab:
                    vec = w + ' ' + vocab[word] + '\n'
                    # print(f'!!found syn {word}')
                    found = True
                    counta += 1
                    break
        
        # find 1st level hyponyms of all synonyms
        if not found:
            for synonym in synonyms:    
                hyponyms = synonym.hyponyms()
                for hyponym in hyponyms:
                    hyponym_word = hyponym.lemmas()[0].name()
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        # print(f'!!found 1st level hyponym {hyponym_word}')
                        found = True
                        countb += 1
                        break
        
        # find 1st level hypernyms of all synonyms
        if not found:
            for synonym in synonyms:
                hypernyms = synonym.hypernyms()
                for hypernym in hypernyms:
                    hypernym_word = hypernym.lemmas()[0].name()
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        # print(f'!!found 1st level hypernym {hypernym_word}')
                        found = True
                        countc += 1
                        break
        
        # find all hyponyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hyponyms = [w for s in synonym.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]
                for hyponym_word in all_hyponyms:
                    stemmed_word = ps.stem(hyponym_word)
                    if hyponym_word in vocab:
                        vec = w + ' ' + vocab[hyponym_word] + '\n'
                        # print(f'!!found hyponym {hyponym_word} for {w}')
                        found = True
                        countb += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        # print(f'!!found hyponym {stemmed_word} for {w}')
                        found = True
                        countb += 1
                        break

        # find all hypernyms recursively, for all synonyms
        if not found:
            for synonym in synonyms:
                all_hypernyms = [w for s in synonym.closure(lambda s:s.hypernyms()) for w in s.lemma_names()]
                for hypernym_word in all_hypernyms:
                    stemmed_word = ps.stem(hypernym_word)
                    if hypernym_word in vocab:
                        vec = w + ' ' + vocab[hypernym_word] + '\n'
                        # print(f'!!found hypernym {hypernym_word} for {w}')
                        found = True
                        countc += 1
                        break 
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        # print(f'!!found hypernym {stemmed_word} for {w}')
                        found = True
                        countc += 1
                        break
        
        '''
        if not found:
            for synonym in synonyms:
                antonyms = synonym.lemmas()[0].antonyms()
                for antonym in antonyms:
                    antonym_word = antonym.name()
                    stemmed_word = ps.stem(antonym_word)
                    if antonym_word in vocab:
                        vec = w + ' ' + vocab[antonym_word] + '\n'
                        # print(f'!!found antonym {antonym_word} for {w}')
                        found = True
                        countz += 1
                        break
                    elif stemmed_word in vocab:
                        vec = w + ' ' + vocab[stemmed_word] + '\n'
                        # print(f'!!found antonym {stemmed_word} for {w}')
                        found = True
                        countz += 1
                        break
        '''          

        if not found:
            vec = w + unk_vector[5:] + ''
            # print(f'syn not found for {w}')
            counte += 1
        print(vec, end = "")

# print(counta, countb, countc, countd, county, countz, counte)

