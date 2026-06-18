from horizon.rhc.tasks.task import Task
import casadi as cs
import numpy as np


class FrameAxisTask(Task):
    def __init__(self, distal_link, frame_axis=None, target_axis=None, base_link=None, *args, **kwargs):
        self.distal_link = distal_link
        self.base_link = 'world' if base_link is None else base_link
        self.frame_axis = self._as_unit_axis(frame_axis, default=[0.0, 0.0, 1.0], name='frame_axis')
        self.target_axis = self._as_unit_axis(target_axis, default=[0.0, 0.0, 1.0], name='target_axis')

        kwargs.setdefault('fun_type', 'residual')
        super().__init__(*args, **kwargs)
        self._createWeightParam()

        if self.fun_type == 'constraint':
            self.instantiator = self.prb.createConstraint
        elif self.fun_type == 'cost':
            self.instantiator = self.prb.createCost
        elif self.fun_type == 'residual':
            self.instantiator = self.prb.createResidual

        self.q = self.prb.getVariables('q')
        self.axis_tgt = self.prb.createParameter(f'{self.name}_{self.distal_link}_axis_tgt', 3)
        self.axis_tgt.assign(self.target_axis)
        self.ref = self.axis_tgt

        fun = self._create_error()
        self.fun = self.instantiator(f'{self.name}_{self.distal_link}_axis', self.weight_param * fun, self.nodes)

    @staticmethod
    def _as_unit_axis(axis, default, name):
        axis = default if axis is None else axis
        axis = np.array(axis, dtype=float).reshape(3)
        norm = np.linalg.norm(axis)
        if norm <= 0.0:
            raise ValueError(f'{name} must have non-zero norm')
        return (axis / norm).reshape(3, 1)

    def _create_error(self):
        fk_distal = self.kin_dyn.fk(self.distal_link)
        distal_pose = fk_distal(q=self.q)
        distal_rot = distal_pose['ee_rot']

        if self.base_link == 'world':
            rel_rot = distal_rot
        else:
            fk_base = self.kin_dyn.fk(self.base_link)
            base_pose = fk_base(q=self.q)
            rel_rot = cs.inv(base_pose['ee_rot']) * distal_rot

        frame_axis = rel_rot @ cs.DM(self.frame_axis)
        return frame_axis - self.axis_tgt

    def setRef(self, ref, nodes=None):
        ref = self._as_unit_axis(ref, default=self.target_axis.reshape(3), name='target_axis')
        self.axis_tgt.assign(ref, nodes)

    def getRef(self):
        return self.axis_tgt

    def setNodes(self, nodes, erasing=True):
        super().setNodes(nodes)
        self.fun.setNodes(nodes, erasing=erasing)
