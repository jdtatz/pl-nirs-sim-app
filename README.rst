################################
pl-nirs-sim-app
################################


Abstract
********

An app to run nirs mcx simulations.

Run
***

Using ``docker run``
====================

Assign an "input" directory to ``/incoming`` and an output directory to ``/outgoing``

.. code-block:: bash

    docker run -v $(pwd)/in:/incoming -v $(pwd)/out:/outgoing   \
            fnndsc/pl-nirs-sim-app nirs_sim_app.py            \
            /incoming /outgoing

This will ...

Make sure that the host ``$(pwd)/out`` directory is world writable!







