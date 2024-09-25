// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.

// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';
import { Icon } from '@iconify/react';

const FeatureList = [
  {
    title: 'Benchmark',
    icon_name: "mdi:ruler",
    description: (
      <>
        First concrete benchmarks for
        LWE attacks on two key real-world
        use cases of LWE: CRYSTALS-KYBER and Homomorphic
        Encryption
      </>
    ),
  },
  {
    title: 'Codebase',
    icon_name: "mdi:code",
    description: (
      <>
        Open-source implementation and evaluation of four concrete LWE
        attacks—uSVP, SALSA, Cool&Cruel, and Dual Hybrid
        MiTM
      </>
    ),
  },
  {
    title: 'Results',
    icon_name: "wpf:unlock-2",
    description: (
      <>
      First ever evaluations of successful secret recovery attacks on
      practical LWE parameters 
      </>
    ),
  },
];

function Feature({icon_name, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Icon icon={icon_name} width={100} color="#2e8555" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
