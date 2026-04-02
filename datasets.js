(function () {
    'use strict';

    const DATASET_CATEGORIES = [
        {
            id: 'dataset-emotion',
            title: 'Emotion & Sentiment',
            countLabel: '7 datasets',
            modalities: ['Text', 'Audio', 'Video'],
            cards: [
                {
                    name: 'MELD',
                    description: 'A large-scale dataset for emotion recognition in conversations. It contains over 13,000 utterances from the TV show Friends, with audio, video, and text.',
                    stats: ['Train: 9,989', 'Val: 1,109', 'Test: 2,610'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'IEMOCAP',
                    description: 'The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is a popular dataset for emotion recognition, containing approximately 12 hours of audiovisual data from ten actors.',
                    stats: ['Train: 5,810', 'Test: 1,623'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'CH-SIMS',
                    description: 'A fine-grained single- and multi-modal sentiment analysis dataset in Chinese. It contains over 2,200 short videos with unimodal and multimodal annotations.',
                    stats: ['Train: 1,368', 'Val: 456', 'Test: 457'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'CH-SIMSv2',
                    description: 'An extended version of CH-SIMS containing over 2,200 short videos with unimodal and multimodal annotations.',
                    stats: ['Train: 2,722', 'Val: 647', 'Test: 1,034'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MVSA-Single',
                    description: 'A Multi-View Sentiment Analysis dataset containing image-text posts from Twitter. The "Single" variant contains posts where the sentiment label is consistent across annotators.',
                    stats: ['Train: 1,555', 'Val: 518', 'Test: 519'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Twitter2015',
                    description: 'Originating from SemEval tasks, these datasets were extended for multimodal aspect-based sentiment analysis, pairing tweets with relevant images.',
                    stats: ['Train: 3,179', 'Val: 1,122', 'Test: 1,037'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Twitter1517',
                    description: 'Extended dataset for multimodal aspect-based sentiment analysis.',
                    stats: ['Train: 3,270', 'Val: 467', 'Test: 935'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        },
        {
            id: 'dataset-social',
            title: 'Social Media',
            countLabel: '6 datasets',
            modalities: ['Text', 'Image'],
            cards: [
                {
                    name: 'MAMI',
                    description: 'A dataset for Misogynistic Meme Detection, containing over 10,000 image-text memes annotated for misogyny and other categories.',
                    stats: ['Train: 9,000', 'Val: 1,000', 'Test: 1,000'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Memotion',
                    description: 'A dataset for analyzing emotions in memes. It contains 10,000 memes annotated for sentiment and three types of emotions (humor, sarcasm, motivation).',
                    stats: ['Train: 5,465', 'Val: 683', 'Test: 683'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MUTE',
                    description: 'Targets troll-like behavior in image/text posts to detect harmful content.',
                    stats: ['Train: 3,365', 'Val: 375', 'Test: 416'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MultiOFF',
                    description: 'Focuses on identifying offensive content and its target in image/text posts.',
                    stats: ['Train: 445', 'Val: 149', 'Test: 149'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MET-Meme (CN)',
                    description: 'A dataset for multimodal metaphor detection in memes (Chinese version).',
                    stats: ['Train: 1,609', 'Val: 229', 'Test: 461'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MET-Meme (EN)',
                    description: 'A dataset for multimodal metaphor detection in memes (English version).',
                    stats: ['Train: 737', 'Val: 105', 'Test: 211'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        },
        {
            id: 'dataset-medical',
            title: 'Healthcare & Medical',
            countLabel: '9 datasets',
            modalities: ['Multi-omics', 'Image', 'Text'],
            cards: [
                {
                    name: 'TCGA-BRCA',
                    description: 'TCGA Breast Cancer (BRCA) multi-omics dataset (gene expression, DNA methylation, copy number variation) for survival prediction.',
                    stats: ['Train: 657', 'Val: 43', 'Test: 175'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'TCGA',
                    description: 'General TCGA multi-omics dataset for survival prediction (dataset selection via IntegrAO).',
                    stats: ['Train: 169', 'Val: 76', 'Test: 61'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'ROSMAP',
                    description: 'Multi-omics data from post-mortem brain tissue for Alzheimer\'s disease research.',
                    stats: ['Train: 194', 'Val: 87', 'Test: 70'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'SIIM-ISIC',
                    description: 'Dermoscopic images of skin lesions with patient-level metadata for melanoma classification (2020 Kaggle challenge).',
                    stats: ['Train: 26,502', 'Val: 3,312', 'Test: 3,312'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Derm7pt',
                    description: 'Multiclass skin lesion classification based on the 7-point checklist. Contains dermoscopic images and semi-quantitative clinical features.',
                    stats: ['Train: 413', 'Val: 203', 'Test: 395'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'GAMMA',
                    description: 'Glaucoma grading dataset containing color fundus images and stereo-pairs of disc photos.',
                    stats: ['Train: 20', 'Val: 10', 'Test: 70'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MIMIC-III',
                    description: 'ICU database with structured data (lab results, vitals) and clinical notes for mortality prediction.',
                    stats: ['Train: 24,462', 'Val: 1,631', 'Test: 6,523'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MIMIC-CXR',
                    description: 'Chest X-ray images and corresponding radiology reports.',
                    stats: ['Train: 5,000', 'Val: 2,942', 'Test: 5,117'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'eICU',
                    description: 'Multi-center ICU database with high-granularity vital sign data and clinical notes.',
                    stats: ['Train: 5,727', 'Val: 382', 'Test: 1,528'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        },
        {
            id: 'dataset-rs',
            title: 'Remote Sensing',
            countLabel: '7 datasets',
            modalities: ['HSI', 'LiDAR', 'SAR'],
            cards: [
                {
                    name: 'Houston2013',
                    description: 'Provides HSI and LiDAR data over the University of Houston campus, covering 15 land use classes (IEEE GRSS Data Fusion Contest).',
                    stats: ['Train: 2,817', 'Test: 12,182'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Houston2018',
                    description: 'A more complex dataset covering 20 urban land use classes (IEEE GRSS Data Fusion Contest).',
                    stats: ['Train: 18,750', 'Test: 2M+'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MUUFL Gulfport',
                    description: 'HSI and LiDAR data collected over the University of Southern Mississippi, Gulfport campus (11 urban land use classes).',
                    stats: ['Train: 1,100', 'Test: 52,587'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Trento',
                    description: 'Covers a rural area south of Trento, Italy. Combines HSI with LiDAR-derived DSM (6 classes).',
                    stats: ['Train: 600', 'Test: 29,614'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'Berlin',
                    description: 'Co-registered HSI and SAR data for Berlin, Germany (8 urban land cover classes).',
                    stats: ['Train: 2,820', 'Test: 461K+'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MDAS (Augsburg)',
                    description: 'Multi-sensor data for urban area classification in Augsburg, Germany, featuring HSI and SAR imagery.',
                    stats: ['Train: 761', 'Test: 77,533'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'ForestNet',
                    description: 'A dataset for wildfire prevention using Sentinel-2 imagery, topography (DSM), and weather data.',
                    stats: ['Train: 1,616', 'Val: 473', 'Test: 668'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        },
        {
            id: 'dataset-vision',
            title: 'Vision & Multimedia',
            countLabel: '6 datasets',
            modalities: ['Image', 'Depth', 'Text'],
            cards: [
                {
                    name: 'MIRFLICKR',
                    description: 'Images from Flickr with associated user-assigned tags, used for retrieval and classification.',
                    stats: ['Train: 14,010', 'Val: 2,001', 'Test: 4,004'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'CUB Image-Caption',
                    description: 'Detailed bird images with rich, descriptive text captions.',
                    stats: ['Train: 82,510', 'Val: 17,680', 'Test: 17,690'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'SUN-RGBD',
                    description: 'Large-scale dataset for indoor scene understanding with RGB-D (color and depth) images.',
                    stats: ['Train: 4,845', 'Test: 4,659'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'NYUDv2',
                    description: 'RGB and Depth images of indoor scenes captured from a Microsoft Kinect.',
                    stats: ['Train: 795', 'Val: 414', 'Test: 654'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'UPMC-Food101',
                    description: 'Food images paired with ingredient lists.',
                    stats: ['Train: 62,971', 'Val: 5,000', 'Test: 22,715'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'MNIST-SVHN',
                    description: 'Synthetic dataset combining MNIST (handwritten digits) and SVHN (street view house numbers).',
                    stats: ['Train: 560,680', 'Test: 100,000'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        },
        {
            id: 'dataset-neuro',
            title: 'Neuromorphic',
            countLabel: '2 datasets',
            modalities: ['Event', 'EEG'],
            cards: [
                {
                    name: 'N-MNIST + N-TIDIGITS',
                    description: 'Neuromorphic versions of MNIST (vision) and TIDIGITS (audio-spoken digits) recorded as event streams.',
                    stats: ['Train: 2,835', 'Val: 603', 'Test: 612'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                },
                {
                    name: 'E-MNIST + EEG',
                    description: 'E-MNIST (handwritten letters/digits) paired with simultaneously recorded EEG brain signals.',
                    stats: ['Train: 468', 'Val: 104', 'Test: 130'],
                    href: 'https://pan.baidu.com/s/11ITMTGO4KCnTLr05dnmThg?pwd=8rc9'
                }
            ]
        }
    ];

    function createBadge(modality) {
        const badge = document.createElement('span');
        badge.className = 'modality-badge';
        badge.textContent = modality;
        return badge;
    }

    function createDatasetCard(card) {
        const link = document.createElement('a');
        link.className = 'dataset-card-link';
        link.href = card.href;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';

        const cardContainer = document.createElement('div');
        cardContainer.className = 'dataset-card-compact';

        const title = document.createElement('h4');
        title.textContent = card.name;
        cardContainer.appendChild(title);

        const description = document.createElement('p');
        description.textContent = card.description;
        cardContainer.appendChild(description);

        const stats = document.createElement('div');
        stats.className = 'card-stats';
        card.stats.forEach(item => {
            const stat = document.createElement('span');
            stat.textContent = item;
            stats.appendChild(stat);
        });
        cardContainer.appendChild(stats);

        link.appendChild(cardContainer);
        return link;
    }

    function createAccordionItem(category) {
        const item = document.createElement('div');
        item.className = 'accordion-item';
        item.id = category.id;

        const header = document.createElement('button');
        header.className = 'accordion-header';
        header.type = 'button';
        header.setAttribute('aria-expanded', 'false');

        const headerContent = document.createElement('div');
        headerContent.className = 'accordion-header-content';

        const title = document.createElement('span');
        title.className = 'accordion-title';
        title.textContent = category.title;

        const count = document.createElement('span');
        count.className = 'accordion-count';
        count.textContent = category.countLabel;

        headerContent.appendChild(title);
        headerContent.appendChild(count);

        const headerRight = document.createElement('div');
        headerRight.className = 'accordion-header-right';

        const badges = document.createElement('div');
        badges.className = 'modality-badges';
        category.modalities.forEach(modality => {
            badges.appendChild(createBadge(modality));
        });

        headerRight.appendChild(badges);
        header.appendChild(headerContent);
        header.appendChild(headerRight);

        const content = document.createElement('div');
        content.className = 'accordion-content';

        const grid = document.createElement('div');
        grid.className = 'dataset-grid-compact';
        category.cards.forEach(card => {
            grid.appendChild(createDatasetCard(card));
        });
        content.appendChild(grid);

        item.appendChild(header);
        item.appendChild(content);
        return item;
    }

    function renderDatasetAccordion(rootId) {
        const mountId = rootId || 'dataset-accordion-root';
        const mount = document.getElementById(mountId);
        if (!mount) return;

        mount.innerHTML = '';

        const fragment = document.createDocumentFragment();
        DATASET_CATEGORIES.forEach(category => {
            fragment.appendChild(createAccordionItem(category));
        });
        mount.appendChild(fragment);
    }

    function createDocCard(card, category) {
        const article = document.createElement('article');
        article.className = 'dataset-doc-card';

        const title = document.createElement('h3');
        title.textContent = card.name;
        article.appendChild(title);

        const details = document.createElement('ul');
        details.className = 'dataset-details';

        const contentItem = document.createElement('li');
        contentItem.innerHTML = '<strong>Data Source and Content:</strong> ';
        contentItem.appendChild(document.createTextNode(card.description));
        details.appendChild(contentItem);

        const modalityItem = document.createElement('li');
        modalityItem.innerHTML = '<strong>Modalities:</strong> ';
        modalityItem.appendChild(document.createTextNode(category.modalities.join(', ')));
        details.appendChild(modalityItem);

        const splitItem = document.createElement('li');
        splitItem.innerHTML = '<strong>Data Splits:</strong> ';
        splitItem.appendChild(document.createTextNode(card.stats.join(', ')));
        details.appendChild(splitItem);

        const linkItem = document.createElement('li');
        linkItem.innerHTML = '<strong>Download:</strong> ';
        const anchor = document.createElement('a');
        anchor.href = card.href;
        anchor.target = '_blank';
        anchor.rel = 'noopener noreferrer';
        anchor.textContent = 'Baidu Netdisk';
        linkItem.appendChild(anchor);
        details.appendChild(linkItem);

        article.appendChild(details);
        return article;
    }

    function createDocSection(category) {
        const section = document.createElement('section');
        section.className = 'doc-section';
        section.id = category.id.replace(/^dataset-/, '');

        const heading = document.createElement('h2');
        heading.textContent = category.title;
        section.appendChild(heading);

        category.cards.forEach(card => {
            section.appendChild(createDocCard(card, category));
        });

        return section;
    }

    function renderDatasetDocumentation(rootId) {
        const mountId = rootId || 'dataset-doc-sections';
        const mount = document.getElementById(mountId);
        if (!mount) return;

        mount.innerHTML = '';

        const fragment = document.createDocumentFragment();
        DATASET_CATEGORIES.forEach(category => {
            fragment.appendChild(createDocSection(category));
        });
        mount.appendChild(fragment);
    }

    function initDatasetViews() {
        renderDatasetAccordion('dataset-accordion-root');
        renderDatasetDocumentation('dataset-doc-sections');
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDatasetViews);
    } else {
        initDatasetViews();
    }

    window.MultiBenchDatasetCatalog = {
        datasetCategories: DATASET_CATEGORIES,
        renderDatasetAccordion,
        renderDatasetDocumentation
    };
})();
